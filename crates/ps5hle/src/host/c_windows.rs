// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The POSIX surface the HLE modules use, on Windows.
//!
//! Modules refer to host C functions as `libc::name`; on Windows they
//! import this module under that name. It re-exports the C runtime (file
//! descriptors, `errno` values, string functions) and adds what the C
//! runtime lacks:
//!
//! * BSD sockets over Winsock. Socket handles are used as descriptors;
//!   `read`, `write`, `close`, `poll`, `ioctl` and `fcntl` recognise them.
//!   Socket options, flags and structures keep their POSIX names and the
//!   values Winsock expects; options Winsock lacks are accepted and
//!   ignored where that is harmless.
//! * Pipes, socket pairs and event descriptors as connected loopback TCP
//!   sockets, so they can be polled together with sockets.
//! * Signal masks (no-ops: Windows has no POSIX signals), clocks, broken
//!   down time, resource limits.
//!
//! Failures set the thread's last error to the C runtime's `errno` value,
//! so `io::Error::last_os_error().raw_os_error()` reads an `errno` as on
//! Linux.

#![allow(non_camel_case_types, non_upper_case_globals, clippy::missing_safety_doc)]

pub use libc::*;

use std::collections::HashMap;
use std::sync::Mutex;
use windows_sys::Win32::Foundation::SetLastError;
use windows_sys::Win32::Networking::WinSock as ws;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

pub type socklen_t = u32;
pub type sa_family_t = u16;
pub type nfds_t = u64;
pub type suseconds_t = i64;
pub type clockid_t = i32;
pub type rlim_t = u64;
pub type sigset_t = u64;
pub type Ioctl = c_ulong;

/// `timeval` with 64-bit fields, as on Linux.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct timeval {
    pub tv_sec: time_t,
    pub tv_usec: suseconds_t,
}

/// Broken-down time with the BSD/glibc `tm_gmtoff` and `tm_zone` fields.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct tm {
    pub tm_sec: c_int,
    pub tm_min: c_int,
    pub tm_hour: c_int,
    pub tm_mday: c_int,
    pub tm_mon: c_int,
    pub tm_year: c_int,
    pub tm_wday: c_int,
    pub tm_yday: c_int,
    pub tm_isdst: c_int,
    pub tm_gmtoff: c_long,
    pub tm_zone: *const c_char,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct in_addr {
    pub s_addr: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct in6_addr {
    pub s6_addr: [u8; 16],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct sockaddr_in {
    pub sin_family: sa_family_t,
    pub sin_port: u16,
    pub sin_addr: in_addr,
    pub sin_zero: [u8; 8],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct sockaddr_in6 {
    pub sin6_family: sa_family_t,
    pub sin6_port: u16,
    pub sin6_flowinfo: u32,
    pub sin6_addr: in6_addr,
    pub sin6_scope_id: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct sockaddr_storage {
    pub ss_family: sa_family_t,
    pub __ss_pad1: [u8; 6],
    pub __ss_align: i64,
    pub __ss_pad2: [u8; 112],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct pollfd {
    pub fd: c_int,
    pub events: c_short,
    pub revents: c_short,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct linger {
    pub l_onoff: c_int,
    pub l_linger: c_int,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ip_mreq {
    pub imr_multiaddr: in_addr,
    pub imr_interface: in_addr,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct iovec {
    pub iov_base: *mut c_void,
    pub iov_len: usize,
}

/// `addrinfo` in the Winsock layout (`ADDRINFOA`).
#[repr(C)]
pub struct addrinfo {
    pub ai_flags: c_int,
    pub ai_family: c_int,
    pub ai_socktype: c_int,
    pub ai_protocol: c_int,
    pub ai_addrlen: usize,
    pub ai_canonname: *mut c_char,
    pub ai_addr: *mut sockaddr,
    pub ai_next: *mut addrinfo,
}

/// Linux `tcp_info` (only its size and the fields read are meaningful).
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct tcp_info {
    pub tcpi_state: u8,
    pub tcpi_ca_state: u8,
    pub tcpi_retransmits: u8,
    pub tcpi_probes: u8,
    pub tcpi_backoff: u8,
    pub tcpi_options: u8,
    pub tcpi_snd_rcv_wscale: u8,
    pub tcpi_flags: u8,
    pub tcpi_rto: u32,
    pub tcpi_ato: u32,
    pub tcpi_snd_mss: u32,
    pub tcpi_rcv_mss: u32,
    pub tcpi_unacked: u32,
    pub tcpi_sacked: u32,
    pub tcpi_lost: u32,
    pub tcpi_retrans: u32,
    pub tcpi_fackets: u32,
    pub tcpi_last_data_sent: u32,
    pub tcpi_last_ack_sent: u32,
    pub tcpi_last_data_recv: u32,
    pub tcpi_last_ack_recv: u32,
    pub tcpi_pmtu: u32,
    pub tcpi_rcv_ssthresh: u32,
    pub tcpi_rtt: u32,
    pub tcpi_rttvar: u32,
    pub tcpi_snd_ssthresh: u32,
    pub tcpi_snd_cwnd: u32,
    pub tcpi_advmss: u32,
    pub tcpi_reordering: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct rlimit {
    pub rlim_cur: rlim_t,
    pub rlim_max: rlim_t,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct rusage {
    pub ru_utime: timeval,
    pub ru_stime: timeval,
    pub ru_maxrss: i64,
    pub ru_ixrss: i64,
    pub ru_idrss: i64,
    pub ru_isrss: i64,
    pub ru_minflt: i64,
    pub ru_majflt: i64,
    pub ru_nswap: i64,
    pub ru_inblock: i64,
    pub ru_oublock: i64,
    pub ru_msgsnd: i64,
    pub ru_msgrcv: i64,
    pub ru_nsignals: i64,
    pub ru_nvcsw: i64,
    pub ru_nivcsw: i64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct statvfs {
    pub f_bsize: c_ulong,
    pub f_frsize: c_ulong,
    pub f_blocks: u64,
    pub f_bfree: u64,
    pub f_bavail: u64,
    pub f_files: u64,
    pub f_ffree: u64,
    pub f_favail: u64,
    pub f_fsid: c_ulong,
    pub f_flag: c_ulong,
    pub f_namemax: c_ulong,
}

// ---------------------------------------------------------------------------
// Constants (Winsock values where Winsock has them)
// ---------------------------------------------------------------------------

pub const AF_UNSPEC: c_int = 0;
pub const AF_UNIX: c_int = 1;
pub const AF_INET: c_int = 2;
pub const AF_INET6: c_int = 23;
pub const SOCK_STREAM: c_int = 1;
pub const SOCK_DGRAM: c_int = 2;
pub const SOCK_RAW: c_int = 3;
pub const SOCK_SEQPACKET: c_int = 5;
/// Accepted and stripped: handles are not inherited unless asked for.
pub const SOCK_CLOEXEC: c_int = 0x0008_0000;
pub const SOCK_NONBLOCK: c_int = 0x0000_0800;
pub const SOL_SOCKET: c_int = 0xFFFF;
pub const SO_DEBUG: c_int = 0x0001;
pub const SO_ACCEPTCONN: c_int = 0x0002;
pub const SO_REUSEADDR: c_int = 0x0004;
pub const SO_KEEPALIVE: c_int = 0x0008;
pub const SO_DONTROUTE: c_int = 0x0010;
pub const SO_BROADCAST: c_int = 0x0020;
pub const SO_LINGER: c_int = 0x0080;
pub const SO_OOBINLINE: c_int = 0x0100;
pub const SO_SNDBUF: c_int = 0x1001;
pub const SO_RCVBUF: c_int = 0x1002;
pub const SO_SNDLOWAT: c_int = 0x1003;
pub const SO_RCVLOWAT: c_int = 0x1004;
pub const SO_SNDTIMEO: c_int = 0x1005;
pub const SO_RCVTIMEO: c_int = 0x1006;
pub const SO_ERROR: c_int = 0x1007;
pub const SO_TYPE: c_int = 0x1008;
/// Not in Winsock: emulated (`SO_DOMAIN`) or ignored (`SO_REUSEPORT`).
pub const SO_DOMAIN: c_int = 0x7001;
pub const SO_REUSEPORT: c_int = 0x7002;
pub const IPPROTO_IP: c_int = 0;
pub const IPPROTO_ICMP: c_int = 1;
pub const IPPROTO_TCP: c_int = 6;
pub const IPPROTO_UDP: c_int = 17;
pub const IPPROTO_IPV6: c_int = 41;
pub const TCP_NODELAY: c_int = 1;
pub const TCP_KEEPIDLE: c_int = 3;
pub const TCP_KEEPCNT: c_int = 16;
pub const TCP_KEEPINTVL: c_int = 17;
/// Not in Winsock: `getsockopt` fails with ENOPROTOOPT.
pub const TCP_INFO: c_int = 0x7003;
pub const IP_TOS: c_int = 3;
pub const IP_TTL: c_int = 4;
pub const IP_MULTICAST_IF: c_int = 9;
pub const IP_MULTICAST_TTL: c_int = 10;
pub const IP_MULTICAST_LOOP: c_int = 11;
pub const IP_ADD_MEMBERSHIP: c_int = 12;
pub const IP_DROP_MEMBERSHIP: c_int = 13;
pub const IPV6_V6ONLY: c_int = 27;
pub const MSG_OOB: c_int = 0x1;
pub const MSG_PEEK: c_int = 0x2;
pub const MSG_DONTROUTE: c_int = 0x4;
pub const MSG_WAITALL: c_int = 0x8;
/// Not in Winsock: handled by `send`/`recv` (`MSG_DONTWAIT` turns the
/// socket non-blocking for the call, `MSG_NOSIGNAL` is the Windows default).
pub const MSG_DONTWAIT: c_int = 0x4000_0000;
pub const MSG_NOSIGNAL: c_int = 0x2000_0000;
pub const SHUT_RD: c_int = 0;
pub const SHUT_WR: c_int = 1;
pub const SHUT_RDWR: c_int = 2;
pub const POLLERR: c_short = 0x0001;
pub const POLLHUP: c_short = 0x0002;
pub const POLLNVAL: c_short = 0x0004;
pub const POLLWRNORM: c_short = 0x0010;
pub const POLLWRBAND: c_short = 0x0020;
pub const POLLRDNORM: c_short = 0x0100;
pub const POLLRDBAND: c_short = 0x0200;
pub const POLLIN: c_short = POLLRDNORM | POLLRDBAND;
pub const POLLPRI: c_short = 0x0400;
pub const POLLOUT: c_short = POLLWRNORM;
/// Not in Winsock: reported with POLLHUP.
pub const POLLRDHUP: c_short = 0x2000;
pub const FIONREAD: Ioctl = 0x4004_667F;
pub const FIONBIO: Ioctl = 0x8004_667E;
/// Not in Winsock: reports 0 bytes queued.
pub const TIOCOUTQ: Ioctl = 0x5411;
pub const AI_PASSIVE: c_int = 0x1;
pub const AI_CANONNAME: c_int = 0x2;
pub const AI_NUMERICHOST: c_int = 0x4;
pub const NI_NUMERICHOST: c_int = 0x2;
pub const NI_NUMERICSERV: c_int = 0x8;
pub const EAI_NONAME: c_int = 11001;
pub const EAI_AGAIN: c_int = 11002;
pub const EAI_FAIL: c_int = 11003;

pub const PROT_NONE: c_int = 0;
pub const PROT_READ: c_int = 1;
pub const PROT_WRITE: c_int = 2;
pub const PROT_EXEC: c_int = 4;

pub const F_GETFD: c_int = 1;
pub const F_SETFD: c_int = 2;
pub const F_GETFL: c_int = 3;
pub const F_SETFL: c_int = 4;
pub const F_GETPIPE_SZ: c_int = 1032;
pub const FD_CLOEXEC: c_int = 1;
pub const O_NONBLOCK: c_int = 0x0000_0800;
pub const O_CLOEXEC: c_int = 0x0008_0000;
pub const O_DIRECTORY: c_int = 0x0001_0000;
pub const O_NOFOLLOW: c_int = 0x0002_0000;
pub const O_PATH: c_int = 0x0020_0000;
pub const O_SYNC: c_int = 0x0010_1000;
pub const O_DSYNC: c_int = 0x0000_1000;

pub const S_IFMT: c_ushort = 0xF000;
pub const S_IFDIR: c_ushort = 0x4000;
pub const S_IFCHR: c_ushort = 0x2000;
pub const S_IFREG: c_ushort = 0x8000;
pub const S_IFIFO: c_ushort = 0x1000;
pub const S_IFSOCK: c_ushort = 0xC000;
pub const S_IFLNK: c_ushort = 0xA000;
pub const S_IFBLK: c_ushort = 0x6000;

pub const SIGHUP: c_int = 1;
pub const SIGQUIT: c_int = 3;
pub const SIGTRAP: c_int = 5;
pub const SIGBUS: c_int = 7;
pub const SIGKILL: c_int = 9;
pub const SIGUSR1: c_int = 10;
pub const SIGUSR2: c_int = 12;
pub const SIGPIPE: c_int = 13;
pub const SIGALRM: c_int = 14;
pub const SIGCHLD: c_int = 17;
pub const SIGCONT: c_int = 18;
pub const SIGSTOP: c_int = 19;
pub const SIGTSTP: c_int = 20;
pub const SIGVTALRM: c_int = 26;
pub const SIGPROF: c_int = 27;
pub const SIG_BLOCK: c_int = 0;
pub const SIG_UNBLOCK: c_int = 1;
pub const SIG_SETMASK: c_int = 2;

pub const CLOCK_REALTIME: clockid_t = 0;
pub const CLOCK_MONOTONIC: clockid_t = 1;
pub const CLOCK_PROCESS_CPUTIME_ID: clockid_t = 2;
pub const CLOCK_THREAD_CPUTIME_ID: clockid_t = 3;
pub const CLOCK_MONOTONIC_RAW: clockid_t = 4;
pub const CLOCK_REALTIME_COARSE: clockid_t = 5;
pub const CLOCK_MONOTONIC_COARSE: clockid_t = 6;
pub const CLOCK_BOOTTIME: clockid_t = 7;

pub const RLIMIT_CPU: c_int = 0;
pub const RLIMIT_FSIZE: c_int = 1;
pub const RLIMIT_DATA: c_int = 2;
pub const RLIMIT_STACK: c_int = 3;
pub const RLIMIT_CORE: c_int = 4;
pub const RLIMIT_RSS: c_int = 5;
pub const RLIMIT_NPROC: c_int = 6;
pub const RLIMIT_NOFILE: c_int = 7;
pub const RLIMIT_MEMLOCK: c_int = 8;
pub const RLIMIT_AS: c_int = 9;
pub const RLIM_INFINITY: rlim_t = u64::MAX;
pub const RUSAGE_SELF: c_int = 0;
pub const RUSAGE_CHILDREN: c_int = -1;
pub const RUSAGE_THREAD: c_int = 1;

pub const _SC_CLK_TCK: c_int = 2;
pub const _SC_OPEN_MAX: c_int = 4;
pub const _SC_PAGESIZE: c_int = 30;
pub const _SC_NPROCESSORS_CONF: c_int = 83;
pub const _SC_NPROCESSORS_ONLN: c_int = 84;
pub const _SC_PHYS_PAGES: c_int = 85;

pub const ST_RDONLY: c_ulong = 1;
pub const ST_NOSUID: c_ulong = 2;
pub const ST_NOEXEC: c_ulong = 8;

pub const GRND_NONBLOCK: c_uint = 1;

// ---------------------------------------------------------------------------
// errno
// ---------------------------------------------------------------------------

/// Record `errno` for this thread (C runtime `errno` and the last error).
pub fn set_errno(e: c_int) {
    // SAFETY: the C runtime's per-thread errno.
    unsafe { *_errno() = e };
    // SAFETY: plain per-thread value.
    unsafe { SetLastError(e as u32) };
}

extern "C" {
    fn _errno() -> *mut c_int;
}

/// Location of this thread's C runtime `errno`.
pub unsafe fn __errno_location() -> *mut c_int {
    _errno()
}

/// The `errno` of a Winsock error.
pub fn errno_of_wsa(e: i32) -> c_int {
    match e {
        ws::WSAEINTR => EINTR,
        ws::WSAEBADF => EBADF,
        ws::WSAEACCES => EACCES,
        ws::WSAEFAULT => EFAULT,
        ws::WSAEINVAL => EINVAL,
        ws::WSAEMFILE => EMFILE,
        ws::WSAEWOULDBLOCK => EAGAIN,
        ws::WSAEINPROGRESS => EINPROGRESS,
        ws::WSAEALREADY => EALREADY,
        ws::WSAENOTSOCK => ENOTSOCK,
        ws::WSAEDESTADDRREQ => EDESTADDRREQ,
        ws::WSAEMSGSIZE => EMSGSIZE,
        ws::WSAEPROTOTYPE => EPROTOTYPE,
        ws::WSAENOPROTOOPT => ENOPROTOOPT,
        ws::WSAEPROTONOSUPPORT | ws::WSAESOCKTNOSUPPORT => EPROTONOSUPPORT,
        ws::WSAEOPNOTSUPP => EOPNOTSUPP,
        ws::WSAEPFNOSUPPORT | ws::WSAEAFNOSUPPORT => EAFNOSUPPORT,
        ws::WSAEADDRINUSE => EADDRINUSE,
        ws::WSAEADDRNOTAVAIL => EADDRNOTAVAIL,
        ws::WSAENETDOWN => ENETDOWN,
        ws::WSAENETUNREACH => ENETUNREACH,
        ws::WSAENETRESET => ENETRESET,
        ws::WSAECONNABORTED => ECONNABORTED,
        ws::WSAECONNRESET => ECONNRESET,
        ws::WSAENOBUFS => ENOBUFS,
        ws::WSAEISCONN => EISCONN,
        ws::WSAENOTCONN => ENOTCONN,
        ws::WSAESHUTDOWN => EPIPE,
        ws::WSAETIMEDOUT => ETIMEDOUT,
        ws::WSAECONNREFUSED => ECONNREFUSED,
        ws::WSAELOOP => ELOOP,
        ws::WSAENAMETOOLONG => ENAMETOOLONG,
        ws::WSAEHOSTDOWN | ws::WSAEHOSTUNREACH => EHOSTUNREACH,
        ws::WSAENOTEMPTY => ENOTEMPTY,
        _ => EIO,
    }
}

fn wsa_fail<T: From<i8>>() -> T {
    // SAFETY: plain per-thread value.
    set_errno(errno_of_wsa(unsafe { ws::WSAGetLastError() }));
    T::from(-1)
}

/// A failed transfer: a timeout of SO_RCVTIMEO/SO_SNDTIMEO is EAGAIN, and
/// a datagram without a destination EDESTADDRREQ, as on POSIX systems.
fn io_fail(fd: c_int) -> isize {
    // SAFETY: plain per-thread value.
    let e = unsafe { ws::WSAGetLastError() };
    let errno = match e {
        ws::WSAETIMEDOUT => EAGAIN,
        ws::WSAENOTCONN if sock_info(fd).is_some_and(|i| i.ty == SOCK_DGRAM) => EDESTADDRREQ,
        _ => errno_of_wsa(e),
    };
    set_errno(errno);
    -1
}

fn fail<T: From<i8>>(e: c_int) -> T {
    set_errno(e);
    T::from(-1)
}

// ---------------------------------------------------------------------------
// Socket table
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Default)]
struct SockInfo {
    family: c_int,
    ty: c_int,
    nonblocking: bool,
    /// An event descriptor: the write end of its pair and the counter.
    event_peer: Option<c_int>,
    /// Read end of a pipe pair or socket pair made over loopback TCP.
    pair: bool,
}

static SOCKETS: Mutex<Option<HashMap<c_int, SockInfo>>> = Mutex::new(None);
static EVENT_COUNT: Mutex<Option<HashMap<c_int, u64>>> = Mutex::new(None);

fn with_sockets<R>(f: impl FnOnce(&mut HashMap<c_int, SockInfo>) -> R) -> R {
    let mut g = SOCKETS.lock().unwrap_or_else(|e| e.into_inner());
    f(g.get_or_insert_with(HashMap::new))
}

fn sock_info(fd: c_int) -> Option<SockInfo> {
    with_sockets(|m| m.get(&fd).copied())
}

fn is_socket(fd: c_int) -> bool {
    sock_info(fd).is_some()
}

fn register(fd: c_int, info: SockInfo) {
    with_sockets(|m| m.insert(fd, info));
}

fn startup() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        // SAFETY: plain initialisation of Winsock 2.2.
        let mut data: ws::WSADATA = unsafe { std::mem::zeroed() };
        unsafe { ws::WSAStartup(0x0202, &mut data) };
    });
}

fn raw(fd: c_int) -> ws::SOCKET {
    fd as u32 as ws::SOCKET
}

fn set_nonblocking(fd: c_int, on: bool) -> c_int {
    let mut v: u32 = on as u32;
    // SAFETY: a socket of the table.
    if unsafe { ws::ioctlsocket(raw(fd), ws::FIONBIO, &mut v) } != 0 {
        return wsa_fail();
    }
    with_sockets(|m| {
        if let Some(i) = m.get_mut(&fd) {
            i.nonblocking = on;
        }
    });
    0
}

// ---------------------------------------------------------------------------
// Sockets
// ---------------------------------------------------------------------------

pub unsafe fn socket(domain: c_int, ty: c_int, protocol: c_int) -> c_int {
    startup();
    let nonblocking = ty & SOCK_NONBLOCK != 0;
    let base = ty & !(SOCK_NONBLOCK | SOCK_CLOEXEC);
    let s = ws::WSASocketW(domain, base, protocol, std::ptr::null(), 0, ws::WSA_FLAG_OVERLAPPED | ws::WSA_FLAG_NO_HANDLE_INHERIT);
    if s == ws::INVALID_SOCKET {
        return wsa_fail();
    }
    let fd = s as c_int;
    register(fd, SockInfo { family: domain, ty: base, ..Default::default() });
    if nonblocking {
        set_nonblocking(fd, true);
    }
    fd
}

pub unsafe fn bind(fd: c_int, addr: *const sockaddr, len: socklen_t) -> c_int {
    if ws::bind(raw(fd), addr as *const ws::SOCKADDR, len as i32) != 0 {
        return wsa_fail();
    }
    0
}

pub unsafe fn connect(fd: c_int, addr: *const sockaddr, len: socklen_t) -> c_int {
    if ws::connect(raw(fd), addr as *const ws::SOCKADDR, len as i32) != 0 {
        return wsa_fail();
    }
    0
}

pub unsafe fn listen(fd: c_int, backlog: c_int) -> c_int {
    if ws::listen(raw(fd), backlog) != 0 {
        return wsa_fail();
    }
    0
}

pub unsafe fn accept4(fd: c_int, addr: *mut sockaddr, len: *mut socklen_t, flags: c_int) -> c_int {
    let s = ws::accept(raw(fd), addr as *mut ws::SOCKADDR, len as *mut i32);
    if s == ws::INVALID_SOCKET {
        return wsa_fail();
    }
    let nfd = s as c_int;
    let family = sock_info(fd).map(|i| i.family).unwrap_or(AF_INET);
    register(nfd, SockInfo { family, ty: SOCK_STREAM, ..Default::default() });
    // Accepted sockets inherit the listener's non-blocking mode on Windows.
    set_nonblocking(nfd, flags & SOCK_NONBLOCK != 0);
    nfd
}

pub unsafe fn accept(fd: c_int, addr: *mut sockaddr, len: *mut socklen_t) -> c_int {
    accept4(fd, addr, len, 0)
}

/// Run `f` with the socket non-blocking when `flags` has MSG_DONTWAIT.
unsafe fn with_flags(fd: c_int, flags: c_int, f: impl FnOnce(c_int) -> isize) -> isize {
    let dontwait = flags & MSG_DONTWAIT != 0;
    let native = flags & !(MSG_DONTWAIT | MSG_NOSIGNAL);
    let was = sock_info(fd).map(|i| i.nonblocking).unwrap_or(false);
    if dontwait && !was {
        set_nonblocking(fd, true);
    }
    let r = f(native);
    if dontwait && !was {
        let e = *_errno();
        set_nonblocking(fd, false);
        set_errno(e);
    }
    r
}

pub unsafe fn send(fd: c_int, buf: *const c_void, len: usize, flags: c_int) -> isize {
    with_flags(fd, flags, |f| {
        let n = ws::send(raw(fd), buf as *const u8, len.min(i32::MAX as usize) as i32, f);
        if n < 0 { io_fail(fd) } else { n as isize }
    })
}

pub unsafe fn recv(fd: c_int, buf: *mut c_void, len: usize, flags: c_int) -> isize {
    with_flags(fd, flags, |f| {
        let n = ws::recv(raw(fd), buf as *mut u8, len.min(i32::MAX as usize) as i32, f);
        if n < 0 { io_fail(fd) } else { n as isize }
    })
}

pub unsafe fn sendto(fd: c_int, buf: *const c_void, len: usize, flags: c_int, addr: *const sockaddr, alen: socklen_t) -> isize {
    with_flags(fd, flags, |f| {
        let n = ws::sendto(raw(fd), buf as *const u8, len.min(i32::MAX as usize) as i32, f, addr as *const ws::SOCKADDR, alen as i32);
        if n < 0 { io_fail(fd) } else { n as isize }
    })
}

pub unsafe fn recvfrom(fd: c_int, buf: *mut c_void, len: usize, flags: c_int, addr: *mut sockaddr, alen: *mut socklen_t) -> isize {
    with_flags(fd, flags, |f| {
        let n = ws::recvfrom(raw(fd), buf as *mut u8, len.min(i32::MAX as usize) as i32, f, addr as *mut ws::SOCKADDR, alen as *mut i32);
        if n < 0 { io_fail(fd) } else { n as isize }
    })
}

pub unsafe fn shutdown(fd: c_int, how: c_int) -> c_int {
    if ws::shutdown(raw(fd), how) != 0 {
        return wsa_fail();
    }
    0
}

pub unsafe fn getsockname(fd: c_int, addr: *mut sockaddr, len: *mut socklen_t) -> c_int {
    if ws::getsockname(raw(fd), addr as *mut ws::SOCKADDR, len as *mut i32) != 0 {
        return wsa_fail();
    }
    0
}

pub unsafe fn getpeername(fd: c_int, addr: *mut sockaddr, len: *mut socklen_t) -> c_int {
    if ws::getpeername(raw(fd), addr as *mut ws::SOCKADDR, len as *mut i32) != 0 {
        return wsa_fail();
    }
    0
}

pub unsafe fn setsockopt(fd: c_int, level: c_int, opt: c_int, val: *const c_void, len: socklen_t) -> c_int {
    if level == SOL_SOCKET && opt == SO_REUSEPORT {
        return 0;
    }
    if level == SOL_SOCKET && (opt == SO_RCVTIMEO || opt == SO_SNDTIMEO) && len as usize >= std::mem::size_of::<timeval>() {
        // Winsock takes milliseconds.
        let tv = &*(val as *const timeval);
        let ms: u32 = (tv.tv_sec.max(0) as u64 * 1000 + tv.tv_usec.max(0) as u64 / 1000).min(u32::MAX as u64) as u32;
        if ws::setsockopt(raw(fd), level, opt, &ms as *const u32 as *const u8, 4) != 0 {
            return wsa_fail();
        }
        return 0;
    }
    if level == SOL_SOCKET && opt == SO_LINGER && len as usize >= std::mem::size_of::<linger>() {
        let l = &*(val as *const linger);
        let w = ws::LINGER { l_onoff: (l.l_onoff != 0) as u16, l_linger: l.l_linger.clamp(0, u16::MAX as i32) as u16 };
        if ws::setsockopt(raw(fd), level, opt, &w as *const ws::LINGER as *const u8, 4) != 0 {
            return wsa_fail();
        }
        return 0;
    }
    if ws::setsockopt(raw(fd), level, opt, val as *const u8, len as i32) != 0 {
        // Options Winsock does not know are ignored, as the console's own
        // network stack ignores unsupported tuning.
        let e = ws::WSAGetLastError();
        if e == ws::WSAENOPROTOOPT || e == ws::WSAEINVAL {
            return 0;
        }
        return wsa_fail();
    }
    0
}

pub unsafe fn getsockopt(fd: c_int, level: c_int, opt: c_int, val: *mut c_void, len: *mut socklen_t) -> c_int {
    if level == SOL_SOCKET && opt == SO_DOMAIN {
        let Some(i) = sock_info(fd) else { return fail(ENOTSOCK) };
        *(val as *mut c_int) = i.family;
        *len = 4;
        return 0;
    }
    if level == IPPROTO_TCP && opt == TCP_INFO {
        return fail(ENOPROTOOPT);
    }
    if level == SOL_SOCKET && (opt == SO_RCVTIMEO || opt == SO_SNDTIMEO) {
        let mut ms: u32 = 0;
        let mut n: i32 = 4;
        if ws::getsockopt(raw(fd), level, opt, &mut ms as *mut u32 as *mut u8, &mut n) != 0 {
            return wsa_fail();
        }
        if (*len as usize) < std::mem::size_of::<timeval>() {
            return fail(EINVAL);
        }
        *(val as *mut timeval) = timeval { tv_sec: (ms / 1000) as time_t, tv_usec: ((ms % 1000) * 1000) as suseconds_t };
        *len = std::mem::size_of::<timeval>() as socklen_t;
        return 0;
    }
    if level == SOL_SOCKET && opt == SO_LINGER {
        let mut w = ws::LINGER { l_onoff: 0, l_linger: 0 };
        let mut n: i32 = 4;
        if ws::getsockopt(raw(fd), level, opt, &mut w as *mut ws::LINGER as *mut u8, &mut n) != 0 {
            return wsa_fail();
        }
        *(val as *mut linger) = linger { l_onoff: w.l_onoff as c_int, l_linger: w.l_linger as c_int };
        *len = std::mem::size_of::<linger>() as socklen_t;
        return 0;
    }
    if ws::getsockopt(raw(fd), level, opt, val as *mut u8, len as *mut i32) != 0 {
        return wsa_fail();
    }
    // Boolean options are reported in one byte by some providers.
    if *len == 1 {
        *(val as *mut c_int) = *(val as *const u8) as c_int;
        *len = 4;
    }
    0
}

/// A connected pair of loopback TCP sockets.
unsafe fn loopback_pair(fds: *mut c_int, nonblocking: bool) -> c_int {
    startup();
    let l = socket(AF_INET, SOCK_STREAM, 0);
    if l < 0 {
        return -1;
    }
    let mut a: sockaddr_in = std::mem::zeroed();
    a.sin_family = AF_INET as u16;
    a.sin_addr.s_addr = u32::from_ne_bytes([127, 0, 0, 1]);
    let mut alen = std::mem::size_of::<sockaddr_in>() as socklen_t;
    let ok = bind(l, &a as *const _ as *const sockaddr, alen) == 0
        && listen(l, 1) == 0
        && getsockname(l, &mut a as *mut _ as *mut sockaddr, &mut alen) == 0;
    if !ok {
        close(l);
        return -1;
    }
    let c = socket(AF_INET, SOCK_STREAM, 0);
    if c < 0 || connect(c, &a as *const _ as *const sockaddr, alen) != 0 {
        close(l);
        if c >= 0 {
            close(c);
        }
        return -1;
    }
    let s = accept4(l, std::ptr::null_mut(), std::ptr::null_mut(), 0);
    close(l);
    if s < 0 {
        close(c);
        return -1;
    }
    let one: c_int = 1;
    setsockopt(c, IPPROTO_TCP, TCP_NODELAY, &one as *const c_int as *const c_void, 4);
    setsockopt(s, IPPROTO_TCP, TCP_NODELAY, &one as *const c_int as *const c_void, 4);
    for fd in [s, c] {
        with_sockets(|m| {
            if let Some(i) = m.get_mut(&fd) {
                i.pair = true;
                i.family = AF_UNIX;
            }
        });
        if nonblocking {
            set_nonblocking(fd, true);
        }
    }
    *fds = s;
    *fds.add(1) = c;
    0
}

/// `socketpair`: a connected loopback pair (stream semantics for every
/// type).
pub unsafe fn socketpair(_domain: c_int, ty: c_int, _protocol: c_int, fds: *mut c_int) -> c_int {
    loopback_pair(fds, ty & SOCK_NONBLOCK != 0)
}

/// `pipe2`: `fds[0]` reads, `fds[1]` writes.
pub unsafe fn pipe2(fds: *mut c_int, flags: c_int) -> c_int {
    loopback_pair(fds, flags & O_NONBLOCK != 0)
}

pub const EFD_CLOEXEC: c_int = O_CLOEXEC;
pub const EFD_NONBLOCK: c_int = O_NONBLOCK;

/// `eventfd`: the read end of a loopback pair; writes add to a counter
/// that a read returns and clears.
pub unsafe fn eventfd(initval: c_uint, flags: c_int) -> c_int {
    let mut fds = [0; 2];
    if loopback_pair(fds.as_mut_ptr(), flags & EFD_NONBLOCK != 0) != 0 {
        return -1;
    }
    with_sockets(|m| {
        if let Some(i) = m.get_mut(&fds[0]) {
            i.event_peer = Some(fds[1]);
        }
    });
    let mut g = EVENT_COUNT.lock().unwrap_or_else(|e| e.into_inner());
    g.get_or_insert_with(HashMap::new).insert(fds[0], initval as u64);
    if initval != 0 {
        ws::send(raw(fds[1]), [1u8].as_ptr(), 1, 0);
    }
    fds[0]
}

// ---------------------------------------------------------------------------
// Descriptors
// ---------------------------------------------------------------------------

pub unsafe fn close(fd: c_int) -> c_int {
    if let Some(i) = with_sockets(|m| m.remove(&fd)) {
        if let Some(peer) = i.event_peer {
            with_sockets(|m| m.remove(&peer));
            ws::closesocket(raw(peer));
            EVENT_COUNT.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(HashMap::new).remove(&fd);
        }
        if ws::closesocket(raw(fd)) != 0 {
            return wsa_fail();
        }
        return 0;
    }
    libc::close(fd)
}

pub unsafe fn read(fd: c_int, buf: *mut c_void, len: usize) -> isize {
    let Some(info) = sock_info(fd) else { return libc::read(fd, buf, len.min(u32::MAX as usize) as c_uint) as isize };
    if info.event_peer.is_some() {
        if len < 8 {
            return fail::<i64>(EINVAL) as isize;
        }
        let mut drain = [0u8; 64];
        let n = ws::recv(raw(fd), drain.as_mut_ptr(), drain.len() as i32, 0);
        if n < 0 {
            return wsa_fail::<i64>() as isize;
        }
        // Drain whatever else is queued without blocking.
        let was = info.nonblocking;
        if !was {
            set_nonblocking(fd, true);
        }
        while ws::recv(raw(fd), drain.as_mut_ptr(), drain.len() as i32, 0) > 0 {}
        if !was {
            set_nonblocking(fd, false);
        }
        let mut g = EVENT_COUNT.lock().unwrap_or_else(|e| e.into_inner());
        let v = g.get_or_insert_with(HashMap::new).insert(fd, 0).unwrap_or(0);
        std::ptr::write_unaligned(buf as *mut u64, v.max(1));
        return 8;
    }
    recv(fd, buf, len, 0)
}

pub unsafe fn write(fd: c_int, buf: *const c_void, len: usize) -> isize {
    let Some(info) = sock_info(fd) else { return libc::write(fd, buf, len.min(u32::MAX as usize) as c_uint) as isize };
    if let Some(peer) = info.event_peer {
        if len < 8 {
            return fail::<i64>(EINVAL) as isize;
        }
        let add = std::ptr::read_unaligned(buf as *const u64);
        {
            let mut g = EVENT_COUNT.lock().unwrap_or_else(|e| e.into_inner());
            *g.get_or_insert_with(HashMap::new).entry(fd).or_insert(0) += add;
        }
        ws::send(raw(peer), [1u8].as_ptr(), 1, 0);
        return 8;
    }
    send(fd, buf, len, 0)
}

pub unsafe fn ioctl<T>(fd: c_int, req: Ioctl, arg: *mut T) -> c_int {
    match req {
        FIONREAD if is_socket(fd) => {
            let mut n: u32 = 0;
            if ws::ioctlsocket(raw(fd), ws::FIONREAD, &mut n) != 0 {
                return wsa_fail();
            }
            *(arg as *mut c_int) = n.min(i32::MAX as u32) as c_int;
            0
        }
        FIONBIO if is_socket(fd) => set_nonblocking(fd, *(arg as *const c_int) != 0),
        TIOCOUTQ if is_socket(fd) => {
            *(arg as *mut c_int) = 0;
            0
        }
        _ => fail(ENOTTY),
    }
}

pub unsafe fn fcntl(fd: c_int, cmd: c_int, arg: c_int) -> c_int {
    match cmd {
        F_GETFL => match sock_info(fd) {
            Some(i) => O_RDWR | if i.nonblocking { O_NONBLOCK } else { 0 },
            None => O_RDWR,
        },
        F_SETFL => {
            if is_socket(fd) {
                set_nonblocking(fd, arg & O_NONBLOCK != 0)
            } else {
                0
            }
        }
        F_GETFD => FD_CLOEXEC,
        F_SETFD => 0,
        F_GETPIPE_SZ => 65536,
        _ => fail(EINVAL),
    }
}

pub unsafe fn poll(fds: *mut pollfd, n: nfds_t, timeout: c_int) -> c_int {
    let fds = std::slice::from_raw_parts_mut(fds, n as usize);
    let mut native: Vec<ws::WSAPOLLFD> = Vec::with_capacity(fds.len());
    let mut index: Vec<usize> = Vec::with_capacity(fds.len());
    let mut invalid = 0;
    for (i, p) in fds.iter_mut().enumerate() {
        p.revents = 0;
        if p.fd < 0 {
            continue;
        }
        if !is_socket(p.fd) {
            p.revents = POLLNVAL;
            invalid += 1;
            continue;
        }
        let events = p.events & (POLLRDNORM | POLLRDBAND | POLLWRNORM);
        native.push(ws::WSAPOLLFD { fd: raw(p.fd), events, revents: 0 });
        index.push(i);
    }
    if invalid > 0 {
        return invalid;
    }
    if native.is_empty() {
        if timeout > 0 {
            std::thread::sleep(std::time::Duration::from_millis(timeout as u64));
        }
        return 0;
    }
    let r = ws::WSAPoll(native.as_mut_ptr(), native.len() as u32, timeout);
    if r < 0 {
        return wsa_fail();
    }
    for (p, &i) in native.iter().zip(&index) {
        let mut rev = p.revents;
        if rev & POLLHUP != 0 && fds[i].events & POLLRDHUP != 0 {
            rev |= POLLRDHUP;
        }
        fds[i].revents = rev;
    }
    r
}

pub unsafe fn getaddrinfo(node: *const c_char, service: *const c_char, hints: *const addrinfo, res: *mut *mut addrinfo) -> c_int {
    startup();
    ws::getaddrinfo(node as *const u8, service as *const u8, hints as *const ws::ADDRINFOA, res as *mut *mut ws::ADDRINFOA)
}

pub unsafe fn freeaddrinfo(res: *mut addrinfo) {
    ws::freeaddrinfo(res as *const ws::ADDRINFOA)
}

pub unsafe fn getnameinfo(
    sa: *const sockaddr,
    salen: socklen_t,
    host: *mut c_char,
    hostlen: u32,
    serv: *mut c_char,
    servlen: u32,
    flags: c_int,
) -> c_int {
    startup();
    ws::getnameinfo(sa as *const ws::SOCKADDR, salen as i32, host as *mut u8, hostlen, serv as *mut u8, servlen, flags)
}

// ---------------------------------------------------------------------------
// Signals (no POSIX signals on Windows: masks are accepted and ignored)
// ---------------------------------------------------------------------------

pub unsafe fn sigemptyset(set: *mut sigset_t) -> c_int {
    *set = 0;
    0
}

pub unsafe fn sigaddset(set: *mut sigset_t, sig: c_int) -> c_int {
    *set |= 1u64 << (sig as u32 & 63);
    0
}

pub unsafe fn sigismember(set: *const sigset_t, sig: c_int) -> c_int {
    (*set >> (sig as u32 & 63) & 1) as c_int
}

pub unsafe fn pthread_sigmask(_how: c_int, _set: *const sigset_t, old: *mut sigset_t) -> c_int {
    if !old.is_null() {
        *old = 0;
    }
    0
}

pub unsafe fn sigtimedwait(_set: *const sigset_t, _info: *mut c_void, _timeout: *const timespec) -> c_int {
    fail(EAGAIN)
}

// ---------------------------------------------------------------------------
// Clocks and time
// ---------------------------------------------------------------------------

fn filetime_ns(f: windows_sys::Win32::Foundation::FILETIME) -> u64 {
    ((f.dwHighDateTime as u64) << 32 | f.dwLowDateTime as u64) * 100
}

pub unsafe fn clock_gettime(clock: clockid_t, ts: *mut timespec) -> c_int {
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};
    let ns: u64 = match clock {
        CLOCK_REALTIME | CLOCK_REALTIME_COARSE => {
            let mut f = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
            windows_sys::Win32::System::SystemInformation::GetSystemTimePreciseAsFileTime(&mut f);
            // 1601-01-01 -> 1970-01-01.
            filetime_ns(f).saturating_sub(11_644_473_600 * 1_000_000_000)
        }
        CLOCK_MONOTONIC | CLOCK_MONOTONIC_RAW | CLOCK_MONOTONIC_COARSE | CLOCK_BOOTTIME => ps5vm::os::monotonic_ns(),
        CLOCK_THREAD_CPUTIME_ID => ps5vm::os::thread_cpu_ns(),
        c if c <= -THREAD_CLOCK_BIAS => match thread_cpu_of((-(c + THREAD_CLOCK_BIAS)) as u32) {
            Some(ns) => ns,
            None => return fail(EINVAL),
        },
        CLOCK_PROCESS_CPUTIME_ID => {
            let z = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
            let (mut a, mut b, mut k, mut u) = (z, z, z, z);
            GetProcessTimes(GetCurrentProcess(), &mut a, &mut b, &mut k, &mut u);
            filetime_ns(k) + filetime_ns(u)
        }
        _ => return fail(EINVAL),
    };
    (*ts).tv_sec = (ns / 1_000_000_000) as time_t;
    (*ts).tv_nsec = (ns % 1_000_000_000) as c_long;
    0
}

pub unsafe fn clock_getres(clock: clockid_t, ts: *mut timespec) -> c_int {
    let ns = match clock {
        CLOCK_REALTIME_COARSE | CLOCK_MONOTONIC_COARSE => 1_000_000,
        CLOCK_PROCESS_CPUTIME_ID | CLOCK_THREAD_CPUTIME_ID => 100,
        0..=7 => 100,
        c if c <= -THREAD_CLOCK_BIAS => 100,
        _ => return fail(EINVAL),
    };
    if !ts.is_null() {
        (*ts).tv_sec = 0;
        (*ts).tv_nsec = ns;
    }
    0
}

/// The C runtime's `struct tm`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct CrtTm([c_int; 9]);

extern "C" {
    fn _localtime64_s(tm: *mut CrtTm, t: *const i64) -> c_int;
    fn _gmtime64_s(tm: *mut CrtTm, t: *const i64) -> c_int;
    fn _mktime64(tm: *mut CrtTm) -> i64;
    fn _mkgmtime64(tm: *mut CrtTm) -> i64;
}

fn from_crt(c: &CrtTm, gmtoff: i64, out: &mut tm) {
    let f = c.0;
    *out = tm {
        tm_sec: f[0],
        tm_min: f[1],
        tm_hour: f[2],
        tm_mday: f[3],
        tm_mon: f[4],
        tm_year: f[5],
        tm_wday: f[6],
        tm_yday: f[7],
        tm_isdst: f[8],
        tm_gmtoff: gmtoff as c_long,
        tm_zone: std::ptr::null(),
    };
}

fn to_crt(t: &tm) -> CrtTm {
    CrtTm([t.tm_sec, t.tm_min, t.tm_hour, t.tm_mday, t.tm_mon, t.tm_year, t.tm_wday, t.tm_yday, t.tm_isdst])
}

pub unsafe fn localtime_r(t: *const time_t, out: *mut tm) -> *mut tm {
    let v = *t;
    let mut c = CrtTm::default();
    if _localtime64_s(&mut c, &v) != 0 {
        return std::ptr::null_mut();
    }
    // Offset east of UTC: the local fields read as UTC, minus the time.
    let mut copy = c;
    let gmtoff = _mkgmtime64(&mut copy) - v;
    from_crt(&c, gmtoff, &mut *out);
    out
}

pub unsafe fn gmtime_r(t: *const time_t, out: *mut tm) -> *mut tm {
    let v = *t;
    let mut c = CrtTm::default();
    if _gmtime64_s(&mut c, &v) != 0 {
        return std::ptr::null_mut();
    }
    from_crt(&c, 0, &mut *out);
    out
}

pub unsafe fn mktime(t: *mut tm) -> time_t {
    let mut c = to_crt(&*t);
    let v = _mktime64(&mut c);
    if v != -1 {
        let mut copy = c;
        let gmtoff = _mkgmtime64(&mut copy) - v;
        from_crt(&c, gmtoff, &mut *t);
    }
    v as time_t
}

pub unsafe fn timegm(t: *mut tm) -> time_t {
    let mut c = to_crt(&*t);
    let v = _mkgmtime64(&mut c);
    if v != -1 {
        from_crt(&c, 0, &mut *t);
    }
    v as time_t
}

// ---------------------------------------------------------------------------
// Anonymous memory
// ---------------------------------------------------------------------------

pub const MAP_SHARED: c_int = 0x01;
pub const MAP_PRIVATE: c_int = 0x02;
pub const MAP_FIXED: c_int = 0x10;
pub const MAP_ANONYMOUS: c_int = 0x20;
pub const MAP_NORESERVE: c_int = 0x4000;
pub const MAP_FIXED_NOREPLACE: c_int = 0x10_0000;
pub const MAP_FAILED: *mut c_void = !0usize as *mut c_void;
pub const MADV_DONTNEED: c_int = 4;

/// Anonymous private mappings only (what the HLE modules map).
pub unsafe fn mmap(addr: *mut c_void, len: usize, prot: c_int, flags: c_int, fd: c_int, _off: off_t) -> *mut c_void {
    if fd != -1 || flags & MAP_ANONYMOUS == 0 {
        set_errno(ENODEV);
        return MAP_FAILED;
    }
    let p = ps5vm::MemProtect::from_prot_bits(prot);
    let (addr, len) = (addr as u64, len as u64);
    let r = if flags & MAP_FIXED_NOREPLACE != 0 {
        ps5vm::os::map_anon_noreplace(addr, len, p).map(|_| addr).ok()
    } else if flags & MAP_FIXED != 0 {
        ps5vm::os::map_anon_fixed(addr, len, p).map(|_| addr).ok()
    } else {
        ps5vm::os::map_anon(len, p)
    };
    match r {
        Some(a) => a as *mut c_void,
        None => {
            set_errno(ENOMEM);
            MAP_FAILED
        }
    }
}

pub unsafe fn munmap(addr: *mut c_void, len: usize) -> c_int {
    ps5vm::os::unmap(addr as u64, len as u64);
    0
}

pub unsafe fn mprotect(addr: *mut c_void, len: usize, prot: c_int) -> c_int {
    if ps5vm::os::protect(addr as u64, len as u64, ps5vm::MemProtect::from_prot_bits(prot)) {
        0
    } else {
        fail(EACCES)
    }
}

/// `MADV_DONTNEED` on private anonymous memory: the pages read as zero.
pub unsafe fn madvise(addr: *mut c_void, len: usize, advice: c_int) -> c_int {
    if advice == MADV_DONTNEED {
        std::ptr::write_bytes(addr as *mut u8, 0, len);
    }
    0
}

/// `process_vm_readv` of this process: copies what is readable.
pub unsafe fn process_vm_readv(_pid: c_int, local: *const iovec, liovcnt: c_ulong, remote: *const iovec, riovcnt: c_ulong, _flags: c_ulong) -> isize {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    if liovcnt != 1 || riovcnt != 1 {
        return fail::<i64>(EINVAL) as isize;
    }
    let (l, r) = (&*local, &*remote);
    let n = l.iov_len.min(r.iov_len);
    let mut got = 0usize;
    let h = windows_sys::Win32::System::Threading::GetCurrentProcess();
    if ReadProcessMemory(h, r.iov_base, l.iov_base, n, &mut got) == 0 && got == 0 {
        return fail::<i64>(EFAULT) as isize;
    }
    got as isize
}

pub type __rlimit_resource_t = c_int;

// ---------------------------------------------------------------------------
// Process
// ---------------------------------------------------------------------------

pub unsafe fn getrlimit(res: c_int, out: *mut rlimit) -> c_int {
    let v = match res {
        RLIMIT_NOFILE => 8192,
        RLIMIT_STACK => 8 << 20,
        _ => RLIM_INFINITY,
    };
    *out = rlimit { rlim_cur: v, rlim_max: v };
    0
}

pub unsafe fn setrlimit(_res: c_int, _limit: *const rlimit) -> c_int {
    0
}

pub unsafe fn getrusage(who: c_int, out: *mut rusage) -> c_int {
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetCurrentThread, GetProcessTimes, GetThreadTimes};
    let z = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
    let (mut a, mut b, mut k, mut u) = (z, z, z, z);
    *out = std::mem::zeroed();
    match who {
        RUSAGE_THREAD => GetThreadTimes(GetCurrentThread(), &mut a, &mut b, &mut k, &mut u),
        RUSAGE_SELF => GetProcessTimes(GetCurrentProcess(), &mut a, &mut b, &mut k, &mut u),
        _ => return 0,
    };
    let tv = |ns: u64| timeval { tv_sec: (ns / 1_000_000_000) as time_t, tv_usec: (ns % 1_000_000_000 / 1000) as suseconds_t };
    (*out).ru_utime = tv(filetime_ns(u));
    (*out).ru_stime = tv(filetime_ns(k));
    0
}

pub unsafe fn sysconf(name: c_int) -> c_long {
    use windows_sys::Win32::System::SystemInformation::{GetSystemInfo, GlobalMemoryStatusEx, MEMORYSTATUSEX, SYSTEM_INFO};
    let mut si: SYSTEM_INFO = std::mem::zeroed();
    GetSystemInfo(&mut si);
    match name {
        _SC_PAGESIZE => si.dwPageSize as c_long,
        _SC_CLK_TCK => 100,
        _SC_OPEN_MAX => 8192,
        _SC_NPROCESSORS_CONF | _SC_NPROCESSORS_ONLN => si.dwNumberOfProcessors as c_long,
        _SC_PHYS_PAGES => {
            let mut m: MEMORYSTATUSEX = std::mem::zeroed();
            m.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
            GlobalMemoryStatusEx(&mut m);
            (m.ullTotalPhys / si.dwPageSize.max(1) as u64) as c_long
        }
        _ => fail::<i32>(EINVAL) as c_long,
    }
}

pub unsafe fn getdtablesize() -> c_int {
    8192
}

pub unsafe fn getppid() -> c_int {
    0
}

pub unsafe fn geteuid() -> c_uint {
    0
}

pub unsafe fn sched_getcpu() -> c_int {
    windows_sys::Win32::System::Threading::GetCurrentProcessorNumber() as c_int
}

#[link(name = "advapi32")]
extern "system" {
    #[link_name = "SystemFunction036"]
    fn RtlGenRandom(buf: *mut c_void, len: u32) -> u8;
}

pub unsafe fn getrandom(buf: *mut c_void, len: usize, _flags: c_uint) -> isize {
    if RtlGenRandom(buf, len.min(u32::MAX as usize) as u32) == 0 {
        return fail::<i64>(EIO) as isize;
    }
    len as isize
}

pub unsafe fn sched_yield() -> c_int {
    std::thread::yield_now();
    0
}

/// The C library's integer parsing (glibc semantics, which the C runtime
/// does not share in corner cases such as `"0x"` without a hex digit):
/// returns (negative, magnitude saturated at u64::MAX, overflowed, end).
unsafe fn parse_integer(s: *const c_char, base: c_int) -> Option<(bool, u64, bool, *const c_char)> {
    if base != 0 && !(2..=36).contains(&base) {
        return None;
    }
    let at = |i: usize| *s.add(i) as u8;
    let mut i = 0;
    while matches!(at(i), b' ' | b'\t' | b'\n' | 0x0B | 0x0C | b'\r') {
        i += 1;
    }
    let neg = at(i) == b'-';
    if matches!(at(i), b'-' | b'+') {
        i += 1;
    }
    let digit = |c: u8| match c {
        b'0'..=b'9' => Some((c - b'0') as u32),
        b'a'..=b'z' => Some((c - b'a') as u32 + 10),
        b'A'..=b'Z' => Some((c - b'A') as u32 + 10),
        _ => None,
    };
    let mut base = base as u32;
    let hex_prefix = at(i) == b'0' && matches!(at(i + 1), b'x' | b'X') && digit(at(i + 2)).is_some_and(|d| d < 16);
    if (base == 0 || base == 16) && hex_prefix {
        base = 16;
        i += 2;
    } else if base == 0 {
        base = if at(i) == b'0' { 8 } else { 10 };
    }
    let start = i;
    let (mut v, mut over) = (0u64, false);
    while let Some(d) = digit(at(i)).filter(|&d| d < base) {
        match v.checked_mul(base as u64).and_then(|x| x.checked_add(d as u64)) {
            Some(n) => v = n,
            None => over = true,
        }
        i += 1;
    }
    if i == start {
        return Some((false, 0, false, s));
    }
    Some((neg, if over { u64::MAX } else { v }, over, s.add(i)))
}

pub unsafe fn strtoull(s: *const c_char, end: *mut *mut c_char, base: c_int) -> u64 {
    let Some((neg, v, over, stop)) = parse_integer(s, base) else {
        set_errno(EINVAL);
        if !end.is_null() {
            *end = s as *mut c_char;
        }
        return 0;
    };
    if !end.is_null() {
        *end = stop as *mut c_char;
    }
    if over {
        set_errno(ERANGE);
        return u64::MAX;
    }
    if neg { v.wrapping_neg() } else { v }
}

pub unsafe fn strtoll(s: *const c_char, end: *mut *mut c_char, base: c_int) -> i64 {
    let Some((neg, v, over, stop)) = parse_integer(s, base) else {
        set_errno(EINVAL);
        if !end.is_null() {
            *end = s as *mut c_char;
        }
        return 0;
    };
    if !end.is_null() {
        *end = stop as *mut c_char;
    }
    if neg {
        if over || v > i64::MIN.unsigned_abs() {
            set_errno(ERANGE);
            return i64::MIN;
        }
        (v as i64).wrapping_neg()
    } else {
        if over || v > i64::MAX as u64 {
            set_errno(ERANGE);
            return i64::MAX;
        }
        v as i64
    }
}

/// `strtol` with a 64-bit `long`, as on Linux.
pub unsafe fn strtol(s: *const c_char, end: *mut *mut c_char, base: c_int) -> i64 {
    strtoll(s, end, base)
}

/// `strtoul` with a 64-bit `long`, as on Linux.
pub unsafe fn strtoul(s: *const c_char, end: *mut *mut c_char, base: c_int) -> u64 {
    strtoull(s, end, base)
}

pub unsafe fn atoi(s: *const c_char) -> c_int {
    strtoll(s, std::ptr::null_mut(), 10) as c_int
}

// ---------------------------------------------------------------------------
// Threads
// ---------------------------------------------------------------------------

/// A host thread: its Windows thread id.
pub type pthread_t = u64;

pub unsafe fn gettid() -> c_int {
    windows_sys::Win32::System::Threading::GetCurrentThreadId() as c_int
}

pub unsafe fn pthread_self() -> pthread_t {
    windows_sys::Win32::System::Threading::GetCurrentThreadId() as pthread_t
}

/// The CPU-time clock of `thread`; only the calling thread's is measured.
/// The CPU-time clock of `thread`: a negative id that encodes its thread
/// id (Linux encodes per-thread clocks as negative ids as well).
pub unsafe fn pthread_getcpuclockid(thread: pthread_t, clock: *mut clockid_t) -> c_int {
    *clock = -(thread as clockid_t) - THREAD_CLOCK_BIAS;
    0
}

const THREAD_CLOCK_BIAS: clockid_t = 16;

/// CPU time of thread `tid` in nanoseconds.
fn thread_cpu_of(tid: u32) -> Option<u64> {
    use windows_sys::Win32::Foundation::{CloseHandle, FILETIME};
    use windows_sys::Win32::System::Threading::{GetThreadTimes, OpenThread, THREAD_QUERY_LIMITED_INFORMATION};
    // SAFETY: a handle we open and close here.
    unsafe {
        let h = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, 0, tid);
        if h.is_null() {
            return None;
        }
        let z = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
        let (mut a, mut b, mut k, mut u) = (z, z, z, z);
        let ok = GetThreadTimes(h, &mut a, &mut b, &mut k, &mut u) != 0;
        CloseHandle(h);
        ok.then(|| filetime_ns(k) + filetime_ns(u))
    }
}

// ---------------------------------------------------------------------------
// Files by descriptor (a Windows handle value, see `host::fs::as_raw_fd`)
// ---------------------------------------------------------------------------

pub const LOCK_SH: c_int = 1;
pub const LOCK_EX: c_int = 2;
pub const LOCK_NB: c_int = 4;
pub const LOCK_UN: c_int = 8;

fn handle(fd: c_int) -> windows_sys::Win32::Foundation::HANDLE {
    fd as isize as windows_sys::Win32::Foundation::HANDLE
}

fn last_error_errno() -> c_int {
    let e = std::io::Error::last_os_error();
    crate::host::errno_of(&e).unwrap_or(EIO)
}

/// `flock` over the whole file with `LockFileEx`.
pub unsafe fn flock(fd: c_int, op: c_int) -> c_int {
    use windows_sys::Win32::Storage::FileSystem::{LockFileEx, UnlockFileEx, LOCKFILE_EXCLUSIVE_LOCK, LOCKFILE_FAIL_IMMEDIATELY};
    let mut ov: windows_sys::Win32::System::IO::OVERLAPPED = std::mem::zeroed();
    if op & LOCK_UN != 0 {
        if UnlockFileEx(handle(fd), 0, u32::MAX, u32::MAX, &mut ov) == 0 {
            // Unlocking a file that holds no lock is not an error.
            return 0;
        }
        return 0;
    }
    let mut flags = 0;
    if op & LOCK_EX != 0 {
        flags |= LOCKFILE_EXCLUSIVE_LOCK;
    }
    if op & LOCK_NB != 0 {
        flags |= LOCKFILE_FAIL_IMMEDIATELY;
    }
    if LockFileEx(handle(fd), flags, 0, u32::MAX, u32::MAX, &mut ov) == 0 {
        let e = std::io::Error::last_os_error().raw_os_error();
        // ERROR_LOCK_VIOLATION / ERROR_IO_PENDING: held by someone else.
        return fail(if matches!(e, Some(33) | Some(997)) { EWOULDBLOCK } else { last_error_errno() });
    }
    0
}

fn filetime_of(tv: &timeval) -> windows_sys::Win32::Foundation::FILETIME {
    let t = (tv.tv_sec + 11_644_473_600) * 10_000_000 + tv.tv_usec * 10;
    windows_sys::Win32::Foundation::FILETIME { dwLowDateTime: t as u32, dwHighDateTime: (t >> 32) as u32 }
}

/// `futimes`: access and modification times; null sets both to now.
pub unsafe fn futimes(fd: c_int, times: *const timeval) -> c_int {
    use windows_sys::Win32::Storage::FileSystem::SetFileTime;
    let (a, m) = if times.is_null() {
        let mut now = windows_sys::Win32::Foundation::FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
        windows_sys::Win32::System::SystemInformation::GetSystemTimeAsFileTime(&mut now);
        (now, now)
    } else {
        (filetime_of(&*times), filetime_of(&*times.add(1)))
    };
    if SetFileTime(handle(fd), std::ptr::null(), &a, &m) == 0 {
        return fail(last_error_errno());
    }
    0
}

/// `utimes` of a path (links are followed).
pub unsafe fn utimes(path: *const c_char, times: *const timeval) -> c_int {
    let Ok(p) = std::ffi::CStr::from_ptr(path).to_str() else { return fail(EINVAL) };
    use std::os::windows::fs::OpenOptionsExt;
    // FILE_WRITE_ATTRIBUTES; backup semantics open directories too.
    match std::fs::OpenOptions::new().access_mode(0x100).custom_flags(0x0200_0000).open(p) {
        Ok(f) => futimes(crate::host::fs::as_raw_fd(&f), times),
        Err(e) => fail(crate::host::errno_of(&e).unwrap_or(EIO)),
    }
}

pub unsafe fn lutimes(path: *const c_char, times: *const timeval) -> c_int {
    utimes(path, times)
}

/// `statvfs` of the volume holding `path`.
pub unsafe fn statvfs(path: *const c_char, out: *mut statvfs) -> c_int {
    use windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;
    let Ok(p) = std::ffi::CStr::from_ptr(path).to_str() else { return fail(EINVAL) };
    let wide: Vec<u16> = std::ffi::OsStr::new(p).encode_wide_nul();
    let (mut avail, mut total, mut free) = (0u64, 0u64, 0u64);
    if GetDiskFreeSpaceExW(wide.as_ptr(), &mut avail, &mut total, &mut free) == 0 {
        return fail(last_error_errno());
    }
    let bs = 4096u64;
    *out = statvfs {
        f_bsize: bs as c_ulong,
        f_frsize: bs as c_ulong,
        f_blocks: total / bs,
        f_bfree: free / bs,
        f_bavail: avail / bs,
        f_files: 1 << 20,
        f_ffree: 1 << 20,
        f_favail: 1 << 20,
        f_fsid: 0,
        f_flag: 0,
        f_namemax: 255,
    };
    0
}

/// `fstatvfs` of the volume holding the open file `fd`.
pub unsafe fn fstatvfs(fd: c_int, out: *mut statvfs) -> c_int {
    use windows_sys::Win32::Storage::FileSystem::GetFinalPathNameByHandleW;
    let mut buf = [0u16; 1024];
    let n = GetFinalPathNameByHandleW(handle(fd), buf.as_mut_ptr(), buf.len() as u32, 0) as usize;
    if n == 0 || n >= buf.len() {
        return fail(EBADF);
    }
    let path = String::from_utf16_lossy(&buf[..n]);
    // `\\?\C:\dir\file` -> `C:\`
    let root: String = path.trim_start_matches(r"\\?\").chars().take(3).collect();
    let Ok(c) = std::ffi::CString::new(root) else { return fail(EINVAL) };
    statvfs(c.as_ptr(), out)
}

trait EncodeWideNul {
    fn encode_wide_nul(&self) -> Vec<u16>;
}

impl EncodeWideNul for std::ffi::OsStr {
    fn encode_wide_nul(&self) -> Vec<u16> {
        use std::os::windows::ffi::OsStrExt;
        self.encode_wide().chain(std::iter::once(0)).collect()
    }
}

// ---------------------------------------------------------------------------
// Values Winsock and the C runtime lack
// ---------------------------------------------------------------------------

pub const ESHUTDOWN: c_int = 2001;
pub const ESOCKTNOSUPPORT: c_int = 2002;
pub const EPFNOSUPPORT: c_int = 2003;
pub const ETOOMANYREFS: c_int = 2004;
pub const EHOSTDOWN: c_int = 2005;
pub const EDQUOT: c_int = 2006;
pub const ESTALE: c_int = 2007;
pub const EREMOTE: c_int = 2008;
pub const EUSERS: c_int = 2009;

/// One value, as on Linux.
pub const EWOULDBLOCK: c_int = EAGAIN;

pub const EAI_BADFLAGS: c_int = 10022;
pub const EAI_FAMILY: c_int = 10047;
pub const EAI_MEMORY: c_int = 8;
pub const EAI_SERVICE: c_int = 10109;
pub const EAI_SOCKTYPE: c_int = 10044;
pub const EAI_OVERFLOW: c_int = 2010;
pub const EAI_SYSTEM: c_int = 2011;
pub const AI_NUMERICSERV: c_int = 0x8;
pub const AI_ALL: c_int = 0x100;
pub const AI_ADDRCONFIG: c_int = 0x400;
pub const AI_V4MAPPED: c_int = 0x800;
pub const NI_NOFQDN: c_int = 0x1;
pub const NI_NAMEREQD: c_int = 0x4;
pub const NI_DGRAM: c_int = 0x10;
pub const NI_MAXHOST: c_int = 1025;
pub const NI_MAXSERV: c_int = 32;
