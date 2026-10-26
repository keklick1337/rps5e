// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceNet — sockets, resolver, memory pools, errno location, byte-order
//! and inet helpers — plus the POSIX socket exports of libkernel.
//!
//! Two independent socket families exist, as on the console:
//!
//! * `sceNetSocket` ids (128..1023, so they fit a guest `fd_set`) wrap host
//!   sockets directly; bind/listen/accept/recv work against the host. UDPP2P
//!   sockets (type 6) are logical sockets multiplexed by a virtual port over
//!   one shared host UDP socket per address and port.
//! * libkernel `socket()` descriptors share the file-descriptor number space.
//!   Each owns a host socket of the same kind from `socket()` on, which
//!   carries its state; the guest-side record keeps what FreeBSD reports
//!   differently (option bits, timeouts as sbintime, shutdown state).
//!
//! Outbound traffic is restricted to loopback unless
//! `RPS5E_NET_REDIRECT=<ip>` is set, in which case every destination is
//! rewritten to that address. Every libSceNet failure returns
//! `SCE_NET_ERROR_*` and stores the matching errno for sceNetErrnoLoc.
//!
//! The resolver is offline: dotted IPv4 literals resolve, names do not.

#[cfg(windows)]
#[allow(unused_imports)]
use crate::host::c as libc;
use std::cell::Cell;
use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard};

use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};

#[path = "net_posix.rs"]
mod posix_ext;

#[path = "net_kmsg.rs"]
mod kmsg;

#[path = "net_scenet.rs"]
mod scenet_ext;

#[cfg(test)]
#[path = "net_ksock_tests.rs"]
mod ksock_tests;

const LIB: &str = "libSceNet";
const LIB_KERNEL: &str = "libkernel";

const fn code(v: u32) -> i32 {
    v as i32
}

const NET_ERROR_RESOLVER_NO_DNS: i32 = code(0x8041_01E1);
const NET_ERROR_NOT_INITIALIZED: i32 = code(0x8041_01C8);

/// Generic kernel error code (0x80020000 | EFAULT).
const KERNEL_EFAULT: HleResult = 0x8002_000Eu32 as i32 as i64;

const MAX_NAME_LENGTH: usize = 256;

/// Orbis address families / socket types / option names (BSD values).
const ORBIS_AF_INET: i32 = 2;
const ORBIS_AF_INET6: i32 = 28;
const SOL_SOCKET: i32 = 0xFFFF;
const SO_REUSEADDR: i32 = 0x0004;
const SO_KEEPALIVE: i32 = 0x0008;
const SO_BROADCAST: i32 = 0x0020;
const SO_REUSEPORT: i32 = 0x0200;
const SO_SNDBUF: i32 = 0x1001;
const SO_RCVBUF: i32 = 0x1002;
const SO_SNDLOWAT: i32 = 0x1003;
const SO_RCVLOWAT: i32 = 0x1004;
const SO_ERROR: i32 = 0x1007;
const SO_NBIO: i32 = 0x1200;
const IPPROTO_IP: i32 = 0;
const IPPROTO_IPV6: i32 = 41;
const IPV6_V6ONLY: i32 = 27;
const IPPROTO_TCP: i32 = 6;
const TCP_NODELAY: i32 = 1;
/// TCP_MAXSEG and the keep-alive tuning options, all kept in the console's
/// own state (never reaching the host stack; behaviour measured on
/// hardware).
const TCP_MAXSEG: i32 = 2;
const TCP_KEEPIDLE: i32 = 0x100;
const TCP_KEEPINTVL: i32 = 0x200;
const TCP_KEEPCNT: i32 = 0x400;
/// TCP_NOPUSH / TCP_NOOPT and the option 0x80: stored as flag bits only.
const TCP_NOPUSH: i32 = 4;
const TCP_NOOPT: i32 = 8;
const TCP_FLAG_0X80: i32 = 0x80;
const IP_TOS: i32 = 3;
const IP_TTL: i32 = 4;
const IP_MULTICAST_IF: i32 = 9;
const IP_MULTICAST_TTL: i32 = 10;
const IP_MULTICAST_LOOP: i32 = 11;
const IP_ADD_MEMBERSHIP: i32 = 12;
const IP_DROP_MEMBERSHIP: i32 = 13;
/// IP_RECVDSTADDR / IP_RECVIF / IP_RECVTTL, kept as bits.
const IP_RECVDSTADDR: i32 = 7;
const IP_RECVIF: i32 = 20;
const IP_RECVTTL: i32 = 65;

const MSG_PEEK: i32 = 0x2;
const MSG_DONTROUTE: i32 = 0x4;
const MSG_WAITALL: i32 = 0x40;
const MSG_DONTWAIT: i32 = 0x80;
const MSG_NOSIGNAL: i32 = 0x20000;

// ---------------------------------------------------------------------------
// Host socket wrapper
// ---------------------------------------------------------------------------

fn last_errno() -> i32 {
    std::io::Error::last_os_error().raw_os_error().unwrap_or(libc::EIO)
}

/// An owned host socket descriptor.
struct HostSocket {
    fd: i32,
}

impl Drop for HostSocket {
    fn drop(&mut self) {
        unsafe { libc::close(self.fd) };
    }
}

impl HostSocket {
    fn new(domain: i32, ty: i32, protocol: i32) -> Result<Self, i32> {
        let fd = unsafe { libc::socket(domain, ty | libc::SOCK_CLOEXEC, protocol) };
        if fd < 0 {
            Err(last_errno())
        } else {
            Ok(Self { fd })
        }
    }

    fn blocking(&self) -> bool {
        let fl = unsafe { libc::fcntl(self.fd, libc::F_GETFL, 0) };
        fl >= 0 && fl & libc::O_NONBLOCK == 0
    }

    fn set_blocking(&self, blocking: bool) {
        unsafe {
            let fl = libc::fcntl(self.fd, libc::F_GETFL, 0);
            if fl >= 0 {
                let nfl = if blocking { fl & !libc::O_NONBLOCK } else { fl | libc::O_NONBLOCK };
                libc::fcntl(self.fd, libc::F_SETFL, nfl);
            }
        }
    }
}

fn set_int_opt(fd: i32, level: i32, opt: i32, value: i32) -> Result<(), i32> {
    let r = unsafe {
        libc::setsockopt(fd, level, opt, &value as *const i32 as *const libc::c_void, 4)
    };
    if r < 0 {
        Err(last_errno())
    } else {
        Ok(())
    }
}

fn get_int_opt(fd: i32, level: i32, opt: i32) -> Result<i32, i32> {
    let mut v: i32 = 0;
    let mut len: libc::socklen_t = 4;
    let r = unsafe { libc::getsockopt(fd, level, opt, &mut v as *mut i32 as *mut libc::c_void, &mut len) };
    if r < 0 {
        Err(last_errno())
    } else {
        Ok(v)
    }
}

fn sockaddr_from(addr: &SocketAddr) -> (libc::sockaddr_storage, libc::socklen_t) {
    let mut ss: libc::sockaddr_storage = unsafe { std::mem::zeroed() };
    let len = match addr {
        SocketAddr::V4(a) => {
            let sin = unsafe { &mut *(&mut ss as *mut _ as *mut libc::sockaddr_in) };
            sin.sin_family = libc::AF_INET as libc::sa_family_t;
            sin.sin_port = a.port().to_be();
            sin.sin_addr.s_addr = u32::from_ne_bytes(a.ip().octets());
            std::mem::size_of::<libc::sockaddr_in>()
        }
        SocketAddr::V6(a) => {
            let sin6 = unsafe { &mut *(&mut ss as *mut _ as *mut libc::sockaddr_in6) };
            sin6.sin6_family = libc::AF_INET6 as libc::sa_family_t;
            sin6.sin6_port = a.port().to_be();
            sin6.sin6_addr.s6_addr = a.ip().octets();
            sin6.sin6_flowinfo = a.flowinfo();
            sin6.sin6_scope_id = a.scope_id();
            std::mem::size_of::<libc::sockaddr_in6>()
        }
    };
    (ss, len as libc::socklen_t)
}

fn sockaddr_to(ss: &libc::sockaddr_storage, len: libc::socklen_t) -> Option<SocketAddr> {
    if len == 0 {
        return None;
    }
    match ss.ss_family as i32 {
        libc::AF_INET => {
            let sin = unsafe { &*(ss as *const _ as *const libc::sockaddr_in) };
            let ip = Ipv4Addr::from(sin.sin_addr.s_addr.to_ne_bytes());
            Some(SocketAddr::new(IpAddr::V4(ip), u16::from_be(sin.sin_port)))
        }
        libc::AF_INET6 => {
            let sin6 = unsafe { &*(ss as *const _ as *const libc::sockaddr_in6) };
            let ip = Ipv6Addr::from(sin6.sin6_addr.s6_addr);
            Some(SocketAddr::V6(std::net::SocketAddrV6::new(ip, u16::from_be(sin6.sin6_port), sin6.sin6_flowinfo, sin6.sin6_scope_id)))
        }
        _ => None,
    }
}

fn local_addr(fd: i32) -> Option<SocketAddr> {
    let mut ss: libc::sockaddr_storage = unsafe { std::mem::zeroed() };
    let mut len = std::mem::size_of::<libc::sockaddr_storage>() as libc::socklen_t;
    let r = unsafe { libc::getsockname(fd, &mut ss as *mut _ as *mut libc::sockaddr, &mut len) };
    if r < 0 {
        None
    } else {
        sockaddr_to(&ss, len)
    }
}

/// IPv4 view of an address: IPv6 addresses keep their low 32 bits.
fn map_to_ipv4(ip: IpAddr) -> [u8; 4] {
    match ip {
        IpAddr::V4(v4) => v4.octets(),
        IpAddr::V6(v6) => {
            let o = v6.octets();
            [o[12], o[13], o[14], o[15]]
        }
    }
}

/// 16-byte Orbis `sockaddr_in`: len, family, port (BE), address, zero pad.
pub(crate) fn guest_sockaddr_in(ip: [u8; 4], port: u16) -> [u8; 16] {
    let mut b = [0u8; 16];
    b[0] = 16;
    b[1] = ORBIS_AF_INET as u8;
    b[2..4].copy_from_slice(&port.to_be_bytes());
    b[4..8].copy_from_slice(&ip);
    b
}

/// Decode an Orbis `sockaddr_in` (family byte must be AF_INET).
#[cfg(test)]
pub(crate) fn parse_guest_sockaddr_in(bytes: &[u8]) -> Option<(Ipv4Addr, u16)> {
    if bytes.len() < 8 || bytes[1] != ORBIS_AF_INET as u8 {
        return None;
    }
    let port = u16::from_be_bytes([bytes[2], bytes[3]]);
    Some((Ipv4Addr::new(bytes[4], bytes[5], bytes[6], bytes[7]), port))
}

/// IPv4 literal parsing with the classic `inet_aton` rules (1-4 parts,
/// decimal / octal / hex components).
pub(crate) fn parse_ipv4_literal(text: &str) -> Option<Ipv4Addr> {
    if text.is_empty() || text.ends_with('.') {
        return None;
    }
    let parts: Vec<&str> = text.split('.').collect();
    if parts.len() > 4 {
        return None;
    }
    let mut values = Vec::with_capacity(4);
    for p in &parts {
        let (digits, radix) = if let Some(h) = p.strip_prefix("0x").or_else(|| p.strip_prefix("0X")) {
            (h, 16)
        } else if p.len() > 1 && p.starts_with('0') {
            (&p[1..], 8)
        } else {
            (*p, 10)
        };
        if digits.is_empty() && radix != 8 {
            return None;
        }
        let v = if digits.is_empty() { 0 } else { u64::from_str_radix(digits, radix).ok()? };
        if !digits.chars().all(|c| c.is_digit(radix)) || v > u32::MAX as u64 {
            return None;
        }
        values.push(v as u32);
    }
    let n = values.len();
    for &v in &values[..n - 1] {
        if v > 0xFF {
            return None;
        }
    }
    let last = values[n - 1];
    let max_last: u64 = match n {
        1 => u32::MAX as u64,
        2 => 0xFF_FFFF,
        3 => 0xFFFF,
        _ => 0xFF,
    };
    if last as u64 > max_last {
        return None;
    }
    let mut addr: u32 = last;
    for (i, &v) in values[..n - 1].iter().enumerate() {
        addr |= v << (24 - 8 * i);
    }
    Some(Ipv4Addr::from(addr))
}

/// Parse an address literal of either family (IPv4 with `inet_aton`
/// rules, IPv6 optionally bracketed).
pub(crate) fn parse_ip_literal(text: &str) -> Option<IpAddr> {
    if let Some(v4) = parse_ipv4_literal(text) {
        return Some(IpAddr::V4(v4));
    }
    let t = text.strip_prefix('[').and_then(|s| s.strip_suffix(']')).unwrap_or(text);
    let t = t.split('%').next().unwrap_or(t);
    t.parse::<Ipv6Addr>().ok().map(IpAddr::V6)
}

/// Host errno -> guest (BSD) errno for socket failures; anything without a
/// guest counterpart is EIO.
fn map_errno(e: i32) -> i32 {
    match e {
        0 => 0,
        libc::EACCES => EACCES_,
        libc::EADDRINUSE => EADDRINUSE_,
        libc::EADDRNOTAVAIL => EADDRNOTAVAIL_,
        libc::EAFNOSUPPORT => EAFNOSUPPORT_,
        libc::EAGAIN => EWOULDBLOCK_,
        libc::EBADF => EBADF_,
        libc::EFAULT => EFAULT_,
        libc::EINTR => EINTR_,
        libc::EINVAL => EINVAL_,
        libc::EISCONN => EISCONN_,
        libc::EMFILE => EMFILE_,
        libc::EMSGSIZE => EMSGSIZE_,
        libc::ENFILE => ENFILE_,
        libc::ENOBUFS => ENOBUFS_,
        libc::ENOMEM => ENOMEM_,
        libc::ENETDOWN => ENETDOWN_,
        libc::ENETRESET => ENETRESET_,
        libc::ENETUNREACH => ENETUNREACH_,
        libc::ENOTCONN => ENOTCONN_,
        libc::ENOTSOCK => ENOTSOCK_,
        libc::ENOPROTOOPT => ENOPROTOOPT_,
        libc::EOPNOTSUPP => EOPNOTSUPP_,
        libc::EPIPE => EPIPE_,
        libc::EPROTONOSUPPORT => EPROTONOSUPPORT_,
        libc::ESHUTDOWN => ESHUTDOWN_,
        libc::ETIMEDOUT => ETIMEDOUT_,
        libc::ECONNABORTED => ECONNABORTED_,
        libc::ECONNREFUSED => ECONNREFUSED_,
        libc::ECONNRESET => ECONNRESET_,
        libc::EDESTADDRREQ => EDESTADDRREQ_,
        libc::EHOSTUNREACH => EHOSTUNREACH_,
        libc::EINPROGRESS => EINPROGRESS_,
        libc::EALREADY => EALREADY_,
        libc::EPERM => EPERM_,
        libc::EPROTOTYPE => EPROTOTYPE_,
        libc::ESOCKTNOSUPPORT => ESOCKTNOSUPPORT_,
        libc::EPFNOSUPPORT => EPFNOSUPPORT_,
        libc::ETOOMANYREFS => ETOOMANYREFS_,
        libc::EHOSTDOWN => EHOSTDOWN_,
        libc::EPROTO => EPROTO_,
        _ => EIO_,
    }
}

/// Retry a syscall returning -1/errno on EINTR.
fn retry<F: FnMut() -> isize>(mut f: F) -> Result<usize, i32> {
    loop {
        let r = f();
        if r >= 0 {
            return Ok(r as usize);
        }
        let e = last_errno();
        if e != libc::EINTR {
            return Err(e);
        }
    }
}

// ---------------------------------------------------------------------------
// libSceNet state
// ---------------------------------------------------------------------------

const POOLS_MAX: usize = 32;
const RESOLVERS_MAX: usize = 32;
const EPOLLS_MAX: usize = 32;
/// sceNet socket ids: 128..1023, so they fit a guest `fd_set`.
const SOCKET_FD_MIN: i32 = 128;
const SOCKET_FD_MAX: i32 = 1024;
/// sceNet epoll ids follow the socket id range.
const EPOLL_ID_BASE: i32 = SOCKET_FD_MAX;

/// Orbis SOCK_DGRAM_P2P: UDP datagrams multiplexed by a virtual port.
const SOCK_DGRAM_P2P: i32 = 6;
/// Host UDP port every UDPP2P socket binds.
const P2P_UDP_PORT: u16 = 3658;
/// SO_SNDTIMEO / SO_RCVTIMEO taking a microsecond count (int); the plain
/// options take a struct timeval.
const SO_SNDTIMEO_USEC: i32 = 0x1105;
const SO_RCVTIMEO_USEC: i32 = 0x1106;
/// struct timeval of the guest: i64 seconds, i64 microseconds.
const GUEST_TIMEVAL_SIZE: u32 = 16;
/// SO_ONESBCAST: broadcast destinations are sent unchanged.
const SO_ONESBCAST: i32 = 0x10000;

// BSD errno values reported through sceNetErrnoLoc.
const ENOENT_: i32 = 2;
const EINTR_: i32 = 4;
const EIO_: i32 = 5;
const EBADF_: i32 = 9;
const ENOMEM_: i32 = 12;
const EACCES_: i32 = 13;
const EFAULT_: i32 = 14;
const EEXIST_: i32 = 17;
const EINVAL_: i32 = 22;
const ENFILE_: i32 = 23;
const EMFILE_: i32 = 24;
const EPIPE_: i32 = 32;
const EDOM_: i32 = 33;
const EWOULDBLOCK_: i32 = 35;
const EINPROGRESS_: i32 = 36;
const EALREADY_: i32 = 37;
const ENOTSOCK_: i32 = 38;
const EDESTADDRREQ_: i32 = 39;
const EMSGSIZE_: i32 = 40;
const ENOPROTOOPT_: i32 = 42;
const EPROTONOSUPPORT_: i32 = 43;
const EOPNOTSUPP_: i32 = 45;
const EAFNOSUPPORT_: i32 = 47;
const EADDRINUSE_: i32 = 48;
const EADDRNOTAVAIL_: i32 = 49;
const ENETDOWN_: i32 = 50;
const ENETUNREACH_: i32 = 51;
const ENETRESET_: i32 = 52;
const ECONNABORTED_: i32 = 53;
const ECONNRESET_: i32 = 54;
const ENOBUFS_: i32 = 55;
const EISCONN_: i32 = 56;
const ENOTCONN_: i32 = 57;
const ESHUTDOWN_: i32 = 58;
const ETIMEDOUT_: i32 = 60;
const ECONNREFUSED_: i32 = 61;
const EHOSTUNREACH_: i32 = 65;

/// SCE_NET_ERROR_EPERM; every errno maps to `NET_ERROR_BASE + errno`.
const NET_ERROR_BASE: i32 = code(0x8041_0100);
const NET_ERROR_RESOLVER_ENOSPACE: i32 = code(0x8041_01DE);

const fn net_error(errno: i32) -> i32 {
    NET_ERROR_BASE + errno
}

/// Per-socket state of a UDPP2P socket.
#[derive(Clone, Copy, Debug, PartialEq)]
struct P2pEndpoint {
    bound: bool,
    /// IPv4 address and UDP port as stored in the sockaddr (network order).
    address: [u8; 4],
    port: [u8; 2],
    /// Virtual port, network order.
    virtual_port: [u8; 2],
    nonblocking: i32,
    broadcast: i32,
    send_buffer: i32,
    receive_buffer: i32,
}

impl Default for P2pEndpoint {
    fn default() -> Self {
        Self {
            bound: false,
            address: [0; 4],
            port: [0; 2],
            virtual_port: [0; 2],
            nonblocking: 0,
            broadcast: 0,
            send_buffer: 9216,
            receive_buffer: 40 * 1024,
        }
    }
}

impl P2pEndpoint {
    fn option(&mut self, option: i32) -> Option<&mut i32> {
        match option {
            SO_BROADCAST => Some(&mut self.broadcast),
            SO_SNDBUF => Some(&mut self.send_buffer),
            SO_RCVBUF => Some(&mut self.receive_buffer),
            SO_NBIO => Some(&mut self.nonblocking),
            _ => None,
        }
    }
}

#[derive(Clone)]
struct SocketSlot {
    transport: Arc<HostSocket>,
    p2p: Option<P2pEndpoint>,
    name: String,
    /// Send / receive timeouts in microseconds as the guest set them (the
    /// host rounds them to its clock).
    timeouts_us: [i32; 2],
    /// The options the console keeps in its own state.
    opts: SockOptState,
    /// The protocol the socket was created with (TCP / UDP for a zero
    /// request on a stream / datagram socket); accept inherits it.
    protocol: i32,
}

#[allow(dead_code)]
struct NetPool {
    name: String,
    size: i32,
}

#[allow(dead_code)]
#[derive(Clone)]
struct Resolver {
    name: String,
    memid: i32,
    last_error: i32,
}

/// A libSceNet epoll object: registered socket ids with their requested
/// events and user data, in registration order.
struct NetEpoll {
    #[allow(dead_code)]
    name: String,
    registrations: Vec<(i32, u32, u64)>,
    /// Set by sceNetEpollAbort; the next wait consumes it and fails with
    /// ECONNABORTED.
    aborted: bool,
}

struct NetState {
    pools: Vec<Option<NetPool>>,
    resolvers: Vec<Option<Resolver>>,
    sockets: HashMap<i32, SocketSlot>,
    epolls: Vec<Option<NetEpoll>>,
}

static NET: LazyLock<Mutex<NetState>> = LazyLock::new(|| {
    Mutex::new(NetState {
        pools: (0..POOLS_MAX).map(|_| None).collect(),
        resolvers: (0..RESOLVERS_MAX).map(|_| None).collect(),
        sockets: HashMap::new(),
        epolls: (0..EPOLLS_MAX).map(|_| None).collect(),
    })
});

/// Signalled whenever sockets or epoll registrations change, so waiting
/// sceNetEpollWait calls re-examine their registrations.
static NET_CHANGED: std::sync::Condvar = std::sync::Condvar::new();

/// The networking module is usable as soon as it is loaded; middleware can
/// create sockets before the application calls sceNetInit.
static INITIALIZED: AtomicBool = AtomicBool::new(true);

fn net() -> MutexGuard<'static, NetState> {
    NET.lock().unwrap_or_else(|p| p.into_inner())
}

impl NetState {
    fn epoll_mut(&mut self, eid: i32) -> Option<&mut NetEpoll> {
        let index = eid.checked_sub(EPOLL_ID_BASE)?;
        if index < 0 {
            return None;
        }
        self.epolls.get_mut(index as usize)?.as_mut()
    }

    fn pool_valid(&self, memid: i32) -> bool {
        memid >= 0 && (memid as usize) < POOLS_MAX && self.pools[memid as usize].is_some()
    }

    fn resolver_mut(&mut self, rid: i32) -> Option<&mut Resolver> {
        if rid < 0 {
            return None;
        }
        self.resolvers.get_mut(rid as usize)?.as_mut()
    }
}

/// Whether `memid` is a live sceNetPoolCreate pool.
pub(crate) fn pool_valid(memid: i32) -> bool {
    net().pool_valid(memid)
}

/// Native socket of `id` for operations that act on the host transport
/// directly; UDPP2P sockets share their transport and are refused.
fn native_socket(id: i32) -> Result<Arc<HostSocket>, i32> {
    let st = net();
    let slot = st.sockets.get(&id).ok_or(EBADF_)?;
    if slot.p2p.is_some() {
        return Err(EOPNOTSUPP_);
    }
    Ok(slot.transport.clone())
}

/// Address family of a host socket.
fn sock_family(fd: i32) -> i32 {
    let mut ss: libc::sockaddr_storage = unsafe { std::mem::zeroed() };
    let mut len = std::mem::size_of::<libc::sockaddr_storage>() as libc::socklen_t;
    // SAFETY: getsockname writes at most `len` bytes into a local.
    if unsafe { libc::getsockname(fd, &mut ss as *mut _ as *mut libc::sockaddr, &mut len) } != 0 {
        return libc::AF_INET;
    }
    ss.ss_family as i32
}

fn socket_slot(id: i32) -> Result<SocketSlot, i32> {
    net().sockets.get(&id).cloned().ok_or(EBADF_)
}

/// Insert a socket under the lowest free id in 128..1023.
fn alloc_socket(slot: SocketSlot) -> Option<i32> {
    let mut st = net();
    let id = (SOCKET_FD_MIN..SOCKET_FD_MAX).find(|id| !st.sockets.contains_key(id))?;
    st.sockets.insert(id, slot);
    Some(id)
}

thread_local! {
    static NET_ERRNO: Cell<u64> = const { Cell::new(0) };
}

/// Per-thread guest slot backing sceNetErrnoLoc.
fn net_errno_address(ctx: &CpuContext) -> u64 {
    NET_ERRNO.with(|c| {
        if c.get() == 0 {
            if let Some(a) = crate::pthread::alloc_guest_block(ctx, 4) {
                c.set(a);
            }
        }
        c.get()
    })
}

fn set_net_error(ctx: &CpuContext, result: i32, errno: i32) -> HleResult {
    let a = net_errno_address(ctx);
    if a != 0 {
        ctx.write_u32(a, errno as u32);
    }
    result as i64
}

/// Result of a libSceNet call: failures also store their errno
/// (`result - SCE_NET_ERROR_EPERM + 1`).
fn finish(ctx: &CpuContext, result: i32) -> HleResult {
    if result < 0 {
        set_net_error(ctx, result, result.wrapping_sub(NET_ERROR_BASE))
    } else {
        result as i64
    }
}

/// Fail with `SCE_NET_ERROR_*` for a BSD errno and set sceNetErrnoLoc.
fn net_fail(ctx: &CpuContext, errno: i32) -> HleResult {
    set_net_error(ctx, net_error(errno), errno)
}

fn socket_result(ctx: &CpuContext, r: Result<i64, i32>) -> HleResult {
    match r {
        Ok(v) => v,
        Err(e) => net_fail(ctx, e),
    }
}

fn net_log(msg: std::fmt::Arguments) {
    log::debug!("[net] {msg}");
}

fn read_name(ctx: &CpuContext, addr: u64, max: usize) -> String {
    ctx.read_cstr_bytes(addr, max).map(|b| String::from_utf8_lossy(&b).into_owned()).unwrap_or_default()
}

/// Guest sockaddr -> IPv4 endpoint: the family byte must be AF_INET and
/// the length cover family, port and address.
fn convert_guest_sockaddr(ctx: &CpuContext, addr: u64, addrlen: u32) -> Result<(Ipv4Addr, u16), i32> {
    if addr == 0 {
        return Err(EFAULT_);
    }
    if addrlen < 2 {
        return Err(EINVAL_);
    }
    let family = ctx.read_bytes(addr + 1, 1).ok_or(EFAULT_)?[0] as i32;
    match family {
        ORBIS_AF_INET => {}
        ORBIS_AF_INET6 => return Err(EOPNOTSUPP_),
        _ => return Err(EAFNOSUPPORT_),
    }
    if addrlen < 8 {
        return Err(EINVAL_);
    }
    let b = ctx.read_bytes(addr, 8).ok_or(EFAULT_)?;
    Ok((Ipv4Addr::new(b[4], b[5], b[6], b[7]), u16::from_be_bytes([b[2], b[3]])))
}

/// Host endpoint -> guest (addr, addrlen*): a 16-byte sockaddr_in; a
/// smaller capacity reports the needed length and fails with EINVAL.
/// Write a host address as a guest sockaddr: as much of it as `*len_addr`
/// holds, with `*len_addr` set to the bytes written. No address (a stream
/// socket's receive) writes nothing and sets the length to 0.
fn convert_host_sockaddr(ctx: &CpuContext, ep: Option<SocketAddr>, addr: u64, len_addr: u64) -> Result<u32, i32> {
    if addr == 0 {
        return Err(EFAULT_);
    }
    let bytes: Vec<u8> = match ep {
        None => {
            return if ctx.write_u32(len_addr, 0) { Ok(0) } else { Err(EFAULT_) };
        }
        Some(SocketAddr::V4(v4)) => guest_sockaddr_in(v4.ip().octets(), v4.port()).to_vec(),
        Some(SocketAddr::V6(v6)) => {
            let mut b = vec![28u8, ORBIS_AF_INET6 as u8];
            b.extend_from_slice(&v6.port().to_be_bytes());
            b.extend_from_slice(&v6.flowinfo().to_ne_bytes());
            b.extend_from_slice(&v6.ip().octets());
            b.extend_from_slice(&v6.scope_id().to_ne_bytes());
            b
        }
    };
    let capacity = ctx.read_u32(len_addr).ok_or(EFAULT_)?;
    let copied = capacity.min(bytes.len() as u32);
    if (copied != 0 && !ctx.write_bytes(addr, &bytes[..copied as usize])) || !ctx.write_u32(len_addr, copied) {
        return Err(EFAULT_);
    }
    Ok(copied)
}

/// Guest message flags -> host flags; unknown bits are EOPNOTSUPP.
fn message_flags(flags: i32) -> Result<i32, i32> {
    let mut host = 0;
    for (guest, h) in [
        (MSG_PEEK, libc::MSG_PEEK),
        (MSG_DONTROUTE, libc::MSG_DONTROUTE),
        (MSG_WAITALL, libc::MSG_WAITALL),
        (MSG_DONTWAIT, libc::MSG_DONTWAIT),
        (MSG_NOSIGNAL, libc::MSG_NOSIGNAL),
    ] {
        if flags & guest != 0 {
            host |= h;
        }
    }
    if flags & !(MSG_PEEK | MSG_DONTROUTE | MSG_WAITALL | MSG_DONTWAIT | MSG_NOSIGNAL) != 0 {
        return Err(EOPNOTSUPP_);
    }
    Ok(host)
}

fn sockaddr_in_host(ip: Ipv4Addr, port: u16) -> (libc::sockaddr_storage, libc::socklen_t) {
    sockaddr_from(&SocketAddr::new(IpAddr::V4(ip), port))
}

// ---------------------------------------------------------------------------
// libSceNet exports
// ---------------------------------------------------------------------------

fn cb_net_init(_ctx: &mut CpuContext) -> HleResult {
    INITIALIZED.store(true, Ordering::Relaxed);
    0
}

fn cb_net_term(_ctx: &mut CpuContext) -> HleResult {
    INITIALIZED.store(false, Ordering::Relaxed);
    let sockets = {
        let mut st = net();
        st.pools.iter_mut().for_each(|p| *p = None);
        st.resolvers.iter_mut().for_each(|r| *r = None);
        st.epolls.iter_mut().for_each(|e| *e = None);
        std::mem::take(&mut st.sockets)
    };
    drop(sockets);
    NET_CHANGED.notify_all();
    0
}

/// sceNetGetMacAddress(SceNetEtherAddr* addr, int flags): no adapter, the
/// address reads as all zeroes.
fn cb_get_mac_address(ctx: &mut CpuContext) -> HleResult {
    let dst = ctx.rdi();
    let flags = ctx.rsi() as i32;
    if dst == 0 || flags != 0 {
        return net_fail(ctx, EINVAL_);
    }
    if !ctx.write_bytes(dst, &[0u8; 6]) {
        return net_fail(ctx, EFAULT_);
    }
    0
}

/// Length of "xx:xx:xx:xx:xx:xx" including its terminator.
const ETHER_ADDRESS_TEXT_SIZE: u64 = 18;

/// "xx:xx:xx:xx:xx:xx" with a terminator.
fn format_ether_address(mac: &[u8]) -> Vec<u8> {
    let mut text = mac.iter().map(|b| format!("{b:02x}")).collect::<Vec<_>>().join(":").into_bytes();
    text.push(0);
    text
}

/// sceNetEtherNtostr(const SceNetEtherAddr* n, char* str, size_t len):
/// any buffer of at least 18 bytes receives the 17 characters and the
/// terminator; anything smaller is rejected without writing.
fn cb_ether_ntostr(ctx: &mut CpuContext) -> HleResult {
    let (src, dst, len) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    if src == 0 || dst == 0 || len < ETHER_ADDRESS_TEXT_SIZE {
        return net_fail(ctx, EINVAL_);
    }
    let Some(mac) = ctx.read_bytes(src, 6) else {
        return net_fail(ctx, EFAULT_);
    };
    if !ctx.write_bytes(dst, &format_ether_address(&mac)) {
        return net_fail(ctx, EFAULT_);
    }
    0
}

/// sceNetSocket(const char* name, int family, int type, int protocol)
fn cb_socket(ctx: &mut CpuContext) -> HleResult {
    if !INITIALIZED.load(Ordering::Relaxed) {
        return finish(ctx, NET_ERROR_NOT_INITIALIZED);
    }
    let family = ctx.rsi() as i32;
    let ty = ctx.rdx() as i32;
    let protocol = ctx.rcx() as i32;
    let domain = match family {
        ORBIS_AF_INET => libc::AF_INET,
        ORBIS_AF_INET6 => libc::AF_INET6,
        _ => return net_fail(ctx, EAFNOSUPPORT_),
    };
    let p2p = ty == SOCK_DGRAM_P2P;
    if p2p && (family != ORBIS_AF_INET || protocol != 0) {
        return net_fail(ctx, EPROTONOSUPPORT_);
    }
    let sock = match HostSocket::new(domain, if p2p { libc::SOCK_DGRAM } else { ty }, protocol) {
        Ok(s) => s,
        Err(e) => return net_fail(ctx, map_errno(e)),
    };
    if p2p {
        // The shared transport never blocks; each logical socket keeps its
        // own blocking mode.
        sock.set_blocking(false);
    }
    let name = read_name(ctx, ctx.rdi(), 32);
    // The protocol the socket reports back: a zero request on a stream or
    // datagram socket means the default one.
    let stored_protocol = if protocol == 0 && ty == SOCK_STREAM_K {
        libc::IPPROTO_TCP as i32
    } else if protocol == 0 && ty == SOCK_DGRAM_K {
        libc::IPPROTO_UDP as i32
    } else {
        protocol
    };
    let slot = SocketSlot { transport: Arc::new(sock), p2p: p2p.then(P2pEndpoint::default), name, timeouts_us: [0; 2], opts: SockOptState::for_family(family), protocol: stored_protocol };
    match alloc_socket(slot) {
        Some(id) => {
            net_log(format_args!("socket.create id={id} family={family} type={ty} protocol={protocol}"));
            id as i64
        }
        None => net_fail(ctx, EMFILE_),
    }
}

/// sceNetSocketClose(int s): the host transport closes with its last user.
fn cb_socket_close(ctx: &mut CpuContext) -> HleResult {
    let id = ctx.rdi() as i32;
    let removed = {
        let mut st = net();
        let removed = st.sockets.remove(&id);
        if removed.is_some() {
            for ep in st.epolls.iter_mut().flatten() {
                ep.registrations.retain(|r| r.0 != id);
            }
        }
        removed
    };
    match removed {
        Some(_) => {
            NET_CHANGED.notify_all();
            0
        }
        // A libkernel socket() descriptor closes through sceNet as well.
        None if kernel_socket_close(id) => 0,
        None => finish(ctx, net_error(EBADF_)),
    }
}

/// Largest single host transfer; stream reads return what fits.
const MAX_TRANSFER: u64 = 4 << 20;

/// sceNetRecv / sceNetRecvfrom (s, buf, len, flags[, addr, addrlen*])
fn net_recv(ctx: &CpuContext, s: i32, buf: u64, len: u64, flags: i32, addr: u64, addrlen: u64) -> Result<i64, i32> {
    if buf == 0 {
        return Err(EFAULT_);
    }
    if addr != 0 && addrlen == 0 {
        return Err(EINVAL_);
    }
    let sock = native_socket(s)?;
    let host_flags = message_flags(flags)?;
    let count = len.min(i32::MAX as u64).min(MAX_TRANSFER) as usize;
    // The outputs are checked before anything is received, so a bad pointer
    // leaves the data queued.
    if addr != 0 {
        if !ps5vm::ranges::accessible(addrlen, 4, true) {
            return Err(EFAULT_);
        }
        if get_int_opt(sock.fd, libc::SOL_SOCKET, libc::SO_TYPE) == Ok(libc::SOCK_DGRAM) {
            let room = ctx.read_u32(addrlen).unwrap_or(0) as u64;
            let family = sock_family(sock.fd);
            let n = room.min(if family == libc::AF_INET6 { 28 } else { 16 });
            if !ps5vm::ranges::accessible(addr, n, true) {
                return Err(EFAULT_);
            }
        }
    }
    if !ps5vm::ranges::accessible(buf, count as u64, true) {
        return Err(EFAULT_);
    }
    let mut payload = vec![0u8; count];
    let mut ss: libc::sockaddr_storage = unsafe { std::mem::zeroed() };
    let mut sslen = std::mem::size_of::<libc::sockaddr_storage>() as libc::socklen_t;
    let received = retry(|| unsafe {
        if addr == 0 {
            libc::recv(sock.fd, payload.as_mut_ptr() as *mut libc::c_void, count, host_flags)
        } else {
            sslen = std::mem::size_of::<libc::sockaddr_storage>() as libc::socklen_t;
            libc::recvfrom(
                sock.fd,
                payload.as_mut_ptr() as *mut libc::c_void,
                count,
                host_flags,
                &mut ss as *mut _ as *mut libc::sockaddr,
                &mut sslen,
            )
        }
    })
    .map_err(map_errno)?;
    if received != 0 && !ctx.write_bytes(buf, &payload[..received]) {
        return Err(EFAULT_);
    }
    if addr != 0 {
        convert_host_sockaddr(ctx, sockaddr_to(&ss, sslen), addr, addrlen)?;
    }
    Ok(received as i64)
}

fn cb_recv(ctx: &mut CpuContext) -> HleResult {
    let r = net_recv(ctx, ctx.rdi() as i32, ctx.rsi(), ctx.rdx(), ctx.rcx() as i32, 0, 0);
    socket_result(ctx, r)
}

fn cb_recvfrom(ctx: &mut CpuContext) -> HleResult {
    let r = net_recv(ctx, ctx.rdi() as i32, ctx.rsi(), ctx.rdx(), ctx.rcx() as i32, ctx.r8(), ctx.r9());
    socket_result(ctx, r)
}

/// Destination for outbound traffic from a sceNet socket after the redirect
/// rule, or the errno when the destination is unreachable while offline.
fn outbound_destination(ip: IpAddr, port: u16) -> Result<SocketAddr, i32> {
    let redirect = net_redirect();
    let ip = redirect.unwrap_or(ip);
    if !outbound_allowed(ip, redirect.is_some()) {
        log::info!("[net] {ip}:{port} unreachable (offline)");
        return Err(ENETUNREACH_);
    }
    Ok(SocketAddr::new(ip, port))
}

/// sceNetSend / sceNetSendto: the library always adds MSG_NOSIGNAL.
fn net_send(ctx: &CpuContext, s: i32, buf: u64, len: u64, flags: i32, addr: u64, addrlen: u32) -> Result<i64, i32> {
    if buf == 0 {
        return Err(EFAULT_);
    }
    let sock = native_socket(s)?;
    let host_flags = message_flags(flags | MSG_NOSIGNAL)?;
    let len = len.min(i32::MAX as u64) as usize;
    let payload = if len == 0 { Vec::new() } else { ctx.read_bytes(buf, len).ok_or(EFAULT_)? };
    let target = if addr == 0 {
        None
    } else {
        let (ip, port) = convert_guest_sockaddr(ctx, addr, addrlen)?;
        Some(sockaddr_from(&outbound_destination(IpAddr::V4(ip), port)?))
    };
    let sent = retry(|| unsafe {
        match &target {
            None => libc::send(sock.fd, payload.as_ptr() as *const libc::c_void, payload.len(), host_flags),
            Some((ss, sslen)) => libc::sendto(
                sock.fd,
                payload.as_ptr() as *const libc::c_void,
                payload.len(),
                host_flags,
                ss as *const _ as *const libc::sockaddr,
                *sslen,
            ),
        }
    })
    .map_err(map_errno)?;
    Ok(sent as i64)
}

/// sceNetSend(s, buf, len, flags)
fn cb_send(ctx: &mut CpuContext) -> HleResult {
    let r = net_send(ctx, ctx.rdi() as i32, ctx.rsi(), ctx.rdx(), ctx.rcx() as i32, 0, 0);
    socket_result(ctx, r)
}

/// sceNetSendto(s, buf, len, flags, to, tolen): NULL `to` sends on the
/// connected peer.
fn cb_sendto(ctx: &mut CpuContext) -> HleResult {
    let r = net_send(ctx, ctx.rdi() as i32, ctx.rsi(), ctx.rdx(), ctx.rcx() as i32, ctx.r8(), ctx.r9() as u32);
    socket_result(ctx, r)
}

fn int_result(r: i32) -> Result<(), i32> {
    if r == 0 { Ok(()) } else { Err(map_errno(last_errno())) }
}

/// Native socket option for a guest (level, option), None if unsupported.
fn host_option(level: i32, option: i32) -> Option<(i32, i32)> {
    Some(match (level, option) {
        (SOL_SOCKET, SO_BROADCAST) => (libc::SOL_SOCKET, libc::SO_BROADCAST),
        (SOL_SOCKET, SO_SNDBUF) => (libc::SOL_SOCKET, libc::SO_SNDBUF),
        (SOL_SOCKET, SO_RCVBUF) => (libc::SOL_SOCKET, libc::SO_RCVBUF),
        (SOL_SOCKET, SO_ERROR) => (libc::SOL_SOCKET, libc::SO_ERROR),
        (SOL_SOCKET, SO_REUSEADDR) => (libc::SOL_SOCKET, libc::SO_REUSEADDR),
        (SOL_SOCKET, SO_REUSEPORT) => (libc::SOL_SOCKET, libc::SO_REUSEPORT),
        (SOL_SOCKET, SO_KEEPALIVE) => (libc::SOL_SOCKET, libc::SO_KEEPALIVE),
        (SOL_SOCKET, SO_OOBINLINE) => (libc::SOL_SOCKET, libc::SO_OOBINLINE),
        (SOL_SOCKET, SO_ACCEPTCONN) => (libc::SOL_SOCKET, libc::SO_ACCEPTCONN),
        (SOL_SOCKET, SO_SNDLOWAT) => (libc::SOL_SOCKET, libc::SO_SNDLOWAT),
        (SOL_SOCKET, SO_RCVLOWAT) => (libc::SOL_SOCKET, libc::SO_RCVLOWAT),
        (SOL_SOCKET, SO_TYPE) => (libc::SOL_SOCKET, libc::SO_TYPE),
        (IPPROTO_TCP, TCP_NODELAY) => (libc::IPPROTO_TCP, libc::TCP_NODELAY),
        (IPPROTO_IPV6, IPV6_V6ONLY) => (libc::IPPROTO_IPV6, libc::IPV6_V6ONLY),
        _ => return None,
    })
}

/// Microseconds of a guest struct timeval for SO_SNDTIMEO / SO_RCVTIMEO:
/// negative fields or 1e6 microseconds and more are EDOM; the seconds are
/// capped at INT_MAX, the result at INT_MAX microseconds.
fn guest_timeval_usec(sec: i64, usec: i64) -> Result<i32, i32> {
    if sec < 0 || !(0..1_000_000).contains(&usec) {
        return Err(EDOM_);
    }
    Ok((sec.min(i32::MAX as i64) * 1_000_000 + usec).min(i32::MAX as i64) as i32)
}

/// Remember the timeout the guest set on socket `s` (still the same socket).
fn store_timeout(s: i32, slot: &SocketSlot, receive: bool, usec: i32) {
    if let Some(cur) = net().sockets.get_mut(&s).filter(|c| Arc::ptr_eq(&c.transport, &slot.transport)) {
        cur.timeouts_us[receive as usize] = usec;
    }
}

fn set_timeval_opt(fd: i32, opt: i32, usec: i32) -> Result<(), i32> {
    let usec = usec.max(0);
    let tv = libc::timeval { tv_sec: (usec / 1_000_000) as libc::time_t, tv_usec: (usec % 1_000_000) as libc::suseconds_t };
    int_result(unsafe {
        libc::setsockopt(fd, libc::SOL_SOCKET, opt, &tv as *const _ as *const libc::c_void, std::mem::size_of::<libc::timeval>() as libc::socklen_t)
    })
}

/// An IPPROTO_TCP option of a sceNet socket: EINVAL on a socket that is
/// not a TCP stream, ENOPROTOOPT for one the console does not have.
fn tcp_level_option(slot: &SocketSlot, option: i32) -> Result<(), i32> {
    let stream = slot.p2p.is_none() && get_int_opt(slot.transport.fd, libc::SOL_SOCKET, libc::SO_TYPE) == Ok(libc::SOCK_STREAM);
    if !stream {
        return Err(EINVAL_);
    }
    if !tcp_option_known(option) {
        return Err(ENOPROTOOPT_);
    }
    Ok(())
}

fn net_getsockopt(ctx: &CpuContext, s: i32, level: i32, option: i32, value_addr: u64, len_addr: u64) -> Result<(), i32> {
    let slot = socket_slot(s)?;
    if value_addr == 0 || len_addr == 0 {
        return Err(EFAULT_);
    }
    let capacity = ctx.read_u32(len_addr).ok_or(EFAULT_)?;
    if level == SOL_SOCKET && option == SO_PROTOCOL {
        // The protocol the socket was created with or accepted from its
        // listener, truncated to the caller's length.
        let n = capacity.min(4) as usize;
        if (n != 0 && !ctx.write_bytes(value_addr, &slot.protocol.to_le_bytes()[..n])) || !ctx.write_u32(len_addr, n as u32) {
            return Err(EFAULT_);
        }
        return Ok(());
    }
    let value = if level == IPPROTO_TCP {
        tcp_level_option(&slot, option)?;
        // TCP_NODELAY is cut to the caller's length (0 to 3 bytes too);
        // the other TCP options need a whole int.
        if capacity < 4 && option != TCP_NODELAY {
            return Err(EINVAL_);
        }
        get_tcp_state_option(slot.transport.fd, &slot.opts, option)?
    } else if level == IPPROTO_IP && slot.p2p.is_none() {
        let fd = slot.transport.fd;
        let ipv4 = get_int_opt(fd, libc::SOL_SOCKET, libc::SO_DOMAIN) == Ok(libc::AF_INET);
        let datagram = get_int_opt(fd, libc::SOL_SOCKET, libc::SO_TYPE) == Ok(libc::SOCK_DGRAM);
        let bytes = get_ip_option(&slot.opts, ipv4, datagram, option, capacity)?;
        let n = capacity.min(bytes.len() as u32);
        if (n != 0 && !ctx.write_bytes(value_addr, &bytes[..n as usize])) || !ctx.write_u32(len_addr, n) {
            return Err(EFAULT_);
        }
        return Ok(());
    } else if let Some(mut p2p) = slot.p2p {
        let v = if level == SOL_SOCKET && kept_or_write_only(option) && option & KEPT_FLAG_OPTIONS != 0 {
            Some(slot.opts.kept_flags & option)
        } else if level == SOL_SOCKET {
            p2p.option(option).map(|v| *v)
        } else {
            None
        };
        let v = v.ok_or(ENOPROTOOPT_)?;
        if capacity < 4 {
            return Err(EINVAL_);
        }
        v
    } else {
        let fd = slot.transport.fd;
        match (level, option) {
            (SOL_SOCKET, SO_SNDTIMEO_USEC | SO_RCVTIMEO_USEC) => {
                if capacity < 4 {
                    return Err(EINVAL_);
                }
                slot.timeouts_us[(option == SO_RCVTIMEO_USEC) as usize]
            }
            // A struct timeval, cut to the caller's length.
            (SOL_SOCKET, SO_SNDTIMEO | SO_RCVTIMEO) => {
                let us = slot.timeouts_us[(option == SO_RCVTIMEO) as usize];
                let mut tv = [0u8; GUEST_TIMEVAL_SIZE as usize];
                tv[..8].copy_from_slice(&(us as i64 / 1_000_000).to_le_bytes());
                tv[8..].copy_from_slice(&(us as i64 % 1_000_000).to_le_bytes());
                let n = capacity.min(GUEST_TIMEVAL_SIZE);
                if (n != 0 && !ctx.write_bytes(value_addr, &tv[..n as usize])) || !ctx.write_u32(len_addr, n) {
                    return Err(EFAULT_);
                }
                return Ok(());
            }
            (SOL_SOCKET, SO_NBIO) => (!slot.transport.blocking()) as i32,
            // The kept flags come back from the console's own state.
            (SOL_SOCKET, _) if kept_or_write_only(option) && option & KEPT_FLAG_OPTIONS != 0 => slot.opts.kept_flags & option,
            _ => {
                let (hl, ho) = host_option(level, option).ok_or(ENOPROTOOPT_)?;
                let v = get_int_opt(fd, hl, ho).map_err(map_errno)?;
                match (level, option) {
                    (SOL_SOCKET, SO_ERROR) if v != 0 => map_errno(v),
                    // A flag reads back as its own bit (FreeBSD's
                    // `so_options & option`).
                    (SOL_SOCKET, SO_KEEPALIVE | SO_BROADCAST | SO_OOBINLINE | SO_ACCEPTCONN) => {
                        if v != 0 {
                            option
                        } else {
                            0
                        }
                    }
                    (SOL_SOCKET, SO_TYPE) => {
                        if v == libc::SOCK_STREAM {
                            1
                        } else {
                            2
                        }
                    }
                    _ => v,
                }
            }
        }
    };
    let n = capacity.min(4) as usize;
    if n != 0 && !ctx.write_bytes(value_addr, &value.to_le_bytes()[..n]) {
        return Err(EFAULT_);
    }
    if !ctx.write_u32(len_addr, n as u32) {
        return Err(EFAULT_);
    }
    Ok(())
}

/// sceNetGetsockopt(s, level, optname, optval, optlen*)
fn cb_getsockopt(ctx: &mut CpuContext) -> HleResult {
    let r = net_getsockopt(ctx, ctx.rdi() as i32, ctx.rsi() as i32, ctx.rdx() as i32, ctx.rcx(), ctx.r8());
    socket_result(ctx, r.map(|_| 0))
}

fn net_setsockopt(ctx: &CpuContext, s: i32, level: i32, option: i32, value_addr: u64, len: u32) -> Result<(), i32> {
    if value_addr == 0 && !(len == 0 && level == SOL_SOCKET && matches!(option, SO_SNDTIMEO | SO_RCVTIMEO)) {
        return Err(EFAULT_);
    }
    let slot = socket_slot(s)?;
    let read_int = || ctx.read_u32(value_addr).map(|v| v as i32).ok_or(EFAULT_);
    if level == IPPROTO_TCP {
        tcp_level_option(&slot, option)?;
        if len < 4 {
            return Err(EINVAL_);
        }
        let number = read_int()?;
        if option == TCP_NODELAY {
            return set_int_opt(slot.transport.fd, libc::IPPROTO_TCP, libc::TCP_NODELAY, (number != 0) as i32).map_err(map_errno);
        }
        let mut st = net();
        let Some(cur) = st.sockets.get_mut(&s).filter(|c| Arc::ptr_eq(&c.transport, &slot.transport)) else {
            return Err(EBADF_);
        };
        return set_tcp_state_option(slot.transport.fd, &mut cur.opts, option, number);
    } else if level == IPPROTO_IP && slot.p2p.is_none() {
        let fd = slot.transport.fd;
        let ipv4 = get_int_opt(fd, libc::SOL_SOCKET, libc::SO_DOMAIN) == Ok(libc::AF_INET);
        let datagram = get_int_opt(fd, libc::SOL_SOCKET, libc::SO_TYPE) == Ok(libc::SOCK_DGRAM);
        let mut st = net();
        let Some(cur) = st.sockets.get_mut(&s).filter(|c| Arc::ptr_eq(&c.transport, &slot.transport)) else {
            return Err(EBADF_);
        };
        return set_ip_option(ctx, fd, &mut cur.opts, ipv4, datagram, option, value_addr, len);
    } else if slot.p2p.is_some() {
        let mut st = net();
        let Some(cur) = st.sockets.get_mut(&s).filter(|c| Arc::ptr_eq(&c.transport, &slot.transport)) else {
            return Err(EBADF_);
        };
        let p2p = cur.p2p.as_mut().ok_or(EBADF_)?;
        if kept_or_write_only(option) {
            if len < 4 {
                return Err(EINVAL_);
            }
            let value = read_int()?;
            let fd = cur.transport.fd;
            return set_kept_flag_option(fd, &mut cur.opts, option, value);
        }
        let target = if level == SOL_SOCKET { p2p.option(option) } else { None };
        let target = target.ok_or(ENOPROTOOPT_)?;
        if len != 4 {
            return Err(EINVAL_);
        }
        let value = read_int()?;
        if option == SO_SNDBUF || option == SO_RCVBUF {
            if value <= 0 {
                return Err(EINVAL_);
            }
            *target = if option == SO_RCVBUF { value.min(512 * 1024) } else { value };
        } else {
            *target = (value != 0) as i32;
        }
        return Ok(());
    }
    let fd = slot.transport.fd;
    match (level, option) {
        (SOL_SOCKET, SO_ONESBCAST) => {
            if len != 4 {
                return Err(EINVAL_);
            }
            if read_int()? != 0 { Ok(()) } else { Err(ENOPROTOOPT_) }
        }
        (SOL_SOCKET, SO_NBIO) if len >= 4 => {
            slot.transport.set_blocking(read_int()? == 0);
            Ok(())
        }
        // An int of microseconds.
        (SOL_SOCKET, SO_SNDTIMEO_USEC | SO_RCVTIMEO_USEC) => {
            if len < 4 {
                return Err(EINVAL_);
            }
            let usec = read_int()?;
            if usec < 0 {
                return Err(EINVAL_);
            }
            let receive = option == SO_RCVTIMEO_USEC;
            set_timeval_opt(fd, if receive { libc::SO_RCVTIMEO } else { libc::SO_SNDTIMEO }, usec)?;
            store_timeout(s, &slot, receive, usec);
            Ok(())
        }
        // A struct timeval {i64 sec, i64 usec}.
        (SOL_SOCKET, SO_SNDTIMEO | SO_RCVTIMEO) => {
            if len < GUEST_TIMEVAL_SIZE {
                return Err(EINVAL_);
            }
            let (Some(sec), Some(usec)) = (ctx.read_u64(value_addr), ctx.read_u64(value_addr + 8)) else {
                return Err(EFAULT_);
            };
            let usec = guest_timeval_usec(sec as i64, usec as i64)?;
            let receive = option == SO_RCVTIMEO;
            set_timeval_opt(fd, if receive { libc::SO_RCVTIMEO } else { libc::SO_SNDTIMEO }, usec)?;
            store_timeout(s, &slot, receive, usec);
            Ok(())
        }
        _ => {
            let (hl, ho) = host_option(level, option).ok_or(ENOPROTOOPT_)?;
            let bytes = if len == 0 { Vec::new() } else { ctx.read_bytes(value_addr, len.min(64) as usize).ok_or(EFAULT_)? };
            int_result(unsafe { libc::setsockopt(fd, hl, ho, bytes.as_ptr() as *const libc::c_void, len.min(64)) })
        }
    }
}

/// sceNetSetsockopt(s, level, optname, optval, optlen)
fn cb_setsockopt(ctx: &mut CpuContext) -> HleResult {
    let s = ctx.rdi() as i32;
    // A libkernel socket() descriptor takes its options through sceNet too
    // (sceNet error codes and errno).
    if !net().sockets.contains_key(&s) && is_kernel_socket(s) {
        let r = kernel_setsockopt(ctx, s, ctx.rsi() as i32, ctx.rdx() as i32, ctx.rcx(), ctx.r8() as u32);
        return socket_result(ctx, r.map(|()| 0));
    }
    let r = net_setsockopt(ctx, s, ctx.rsi() as i32, ctx.rdx() as i32, ctx.rcx(), ctx.r8() as u32);
    socket_result(ctx, r.map(|_| 0))
}

/// POSIX setsockopt: kernel socket descriptors first, then sceNet ids.
fn cb_posix_setsockopt(ctx: &mut CpuContext) -> HleResult {
    if is_kernel_socket(ctx.rdi() as i32) {
        let r = kernel_setsockopt(ctx, ctx.rdi() as i32, ctx.rsi() as i32, ctx.rdx() as i32, ctx.rcx(), ctx.r8() as u32);
        posix_socket_result(ctx, r.map(|()| 0))
    } else {
        cb_setsockopt(ctx)
    }
}

/// POSIX getsockopt: kernel socket descriptors first, then sceNet ids.
fn cb_posix_getsockopt(ctx: &mut CpuContext) -> HleResult {
    if is_kernel_socket(ctx.rdi() as i32) {
        let r = kernel_getsockopt(ctx, ctx.rdi() as i32, ctx.rsi() as i32, ctx.rdx() as i32, ctx.rcx(), ctx.r8());
        posix_socket_result(ctx, r.map(|()| 0))
    } else {
        cb_getsockopt(ctx)
    }
}

/// POSIX send(s, buf, len, flags): kernel socket descriptors first, then
/// sceNet ids.
fn cb_posix_send(ctx: &mut CpuContext) -> HleResult {
    if is_kernel_socket(ctx.rdi() as i32) {
        let r = send_core(ctx, ctx.rdi() as i32, ctx.rsi(), ctx.rdx(), ctx.rcx() as i32, 0, 0);
        posix_socket_result(ctx, r)
    } else {
        cb_send(ctx)
    }
}

/// POSIX sendto: kernel socket descriptors first, then sceNet ids.
fn cb_posix_sendto(ctx: &mut CpuContext) -> HleResult {
    if is_kernel_socket(ctx.rdi() as i32) {
        kernel_sendto(ctx)
    } else {
        cb_sendto(ctx)
    }
}

/// POSIX recvfrom: kernel socket descriptors first, then sceNet ids.
fn cb_posix_recvfrom(ctx: &mut CpuContext) -> HleResult {
    if is_kernel_socket(ctx.rdi() as i32) {
        kernel_recvfrom(ctx)
    } else {
        cb_recvfrom(ctx)
    }
}

/// Text form of a raw address, None for unsupported families.
pub(crate) fn format_address(family: i32, raw: &[u8]) -> Option<String> {
    match family {
        ORBIS_AF_INET if raw.len() >= 4 => Some(Ipv4Addr::new(raw[0], raw[1], raw[2], raw[3]).to_string()),
        ORBIS_AF_INET6 if raw.len() >= 16 => {
            let mut o = [0u8; 16];
            o.copy_from_slice(&raw[..16]);
            Some(Ipv6Addr::from(o).to_string())
        }
        _ => None,
    }
}

/// inet_ntop(af, src, dst, size) -> dst or NULL. The terminator must fit.
fn cb_inet_ntop(ctx: &mut CpuContext) -> HleResult {
    let family = ctx.rdi() as i32;
    let src = ctx.rsi();
    let dst = ctx.rdx();
    let size = ctx.rcx() as u32;
    if src == 0 || dst == 0 || size == 0 {
        return 0;
    }
    let len = match family {
        ORBIS_AF_INET => 4,
        ORBIS_AF_INET6 => 16,
        _ => return 0,
    };
    let Some(raw) = ctx.read_bytes(src, len) else {
        return 0;
    };
    let Some(text) = format_address(family, &raw) else {
        return 0;
    };
    if text.len() + 1 > size as usize {
        return 0;
    }
    let mut out = text.into_bytes();
    out.push(0);
    if !ctx.write_bytes(dst, &out) {
        return 0;
    }
    dst as i64
}

fn net_bind(ctx: &CpuContext, s: i32, addr: u64, addrlen: u32) -> Result<(), i32> {
    if addr == 0 {
        return Err(EFAULT_);
    }
    let slot = socket_slot(s)?;
    let (ip, port) = convert_guest_sockaddr(ctx, addr, addrlen)?;
    if slot.p2p.is_none() {
        let (ss, sslen) = sockaddr_in_host(ip, port);
        int_result(unsafe { libc::bind(slot.transport.fd, &ss as *const _ as *const libc::sockaddr, sslen) })?;
        net_log(format_args!("socket.bind id={s} {ip}:{port}"));
        return Ok(());
    }
    if addrlen < 16 {
        return Err(EINVAL_);
    }
    let vport_raw = ctx.read_bytes(addr + 8, 2).ok_or(EFAULT_)?;
    let virtual_port = [vport_raw[0], vport_raw[1]];
    let requested = u16::from_be_bytes(virtual_port);
    if requested == 0 {
        return Err(EOPNOTSUPP_);
    }
    if port != P2P_UDP_PORT {
        return Err(EINVAL_);
    }
    if requested == 5353 || requested >= 32768 {
        return Err(EACCES_);
    }
    let (address, port_raw) = (ip.octets(), port.to_be_bytes());
    let mut st = net();
    match st.sockets.get(&s) {
        Some(cur) if Arc::ptr_eq(&cur.transport, &slot.transport) => {
            if cur.p2p.is_some_and(|p| p.bound) {
                return Err(EINVAL_);
            }
        }
        _ => return Err(EBADF_),
    }
    let same_endpoint = |p: &P2pEndpoint| p.bound && p.port == port_raw && p.address == address;
    if st.sockets.values().any(|o| o.p2p.as_ref().is_some_and(|p| same_endpoint(p) && p.virtual_port == virtual_port)) {
        return Err(EADDRINUSE_);
    }
    let shared = st.sockets.values().find(|o| o.p2p.as_ref().is_some_and(same_endpoint)).map(|o| o.transport.clone());
    let transport = match shared {
        Some(t) => t,
        None => {
            let (ss, sslen) = sockaddr_in_host(ip, port);
            int_result(unsafe { libc::bind(slot.transport.fd, &ss as *const _ as *const libc::sockaddr, sslen) })?;
            slot.transport.clone()
        }
    };
    let cur = st.sockets.get_mut(&s).ok_or(EBADF_)?;
    cur.transport = transport;
    let p = cur.p2p.as_mut().ok_or(EBADF_)?;
    p.bound = true;
    p.address = address;
    p.port = port_raw;
    p.virtual_port = virtual_port;
    net_log(format_args!("socket.bind id={s} UDPP2P port={port} virtual port={requested}"));
    Ok(())
}

/// sceNetBind(s, addr, addrlen)
fn cb_bind(ctx: &mut CpuContext) -> HleResult {
    let r = net_bind(ctx, ctx.rdi() as i32, ctx.rsi(), ctx.rdx() as u32);
    socket_result(ctx, r.map(|_| 0))
}

/// sceNetConnect(s, addr, addrlen): host connect, subject to the outbound
/// policy (loopback only unless RPS5E_NET_REDIRECT is set).
fn net_connect(ctx: &CpuContext, s: i32, addr: u64, addrlen: u32) -> Result<(), i32> {
    if addr == 0 {
        return Err(EFAULT_);
    }
    let sock = native_socket(s)?;
    let (ip, port) = convert_guest_sockaddr(ctx, addr, addrlen)?;
    let dest = outbound_destination(IpAddr::V4(ip), port)?;
    let (ss, sslen) = sockaddr_from(&dest);
    int_result(unsafe { libc::connect(sock.fd, &ss as *const _ as *const libc::sockaddr, sslen) })?;
    net_log(format_args!("socket.connect id={s} {dest}"));
    Ok(())
}

fn cb_connect(ctx: &mut CpuContext) -> HleResult {
    let r = net_connect(ctx, ctx.rdi() as i32, ctx.rsi(), ctx.rdx() as u32);
    socket_result(ctx, r.map(|_| 0))
}

/// sceNetListen(s, backlog)
fn cb_listen(ctx: &mut CpuContext) -> HleResult {
    let r = native_socket(ctx.rdi() as i32).and_then(|sock| int_result(unsafe { libc::listen(sock.fd, ctx.rsi() as i32) }));
    socket_result(ctx, r.map(|_| 0))
}

/// sceNetAccept(s, addr, addrlen*) -> new socket id.
fn net_accept(ctx: &CpuContext, s: i32, addr: u64, addrlen: u64) -> Result<i64, i32> {
    let sock = native_socket(s)?;
    // The accepted socket inherits the listener's non-blocking mode, as on
    // FreeBSD (Linux accept does not carry O_NONBLOCK over).
    let accept_flags = libc::SOCK_CLOEXEC | if sock.blocking() { 0 } else { libc::SOCK_NONBLOCK };
    let mut ss: libc::sockaddr_storage = unsafe { std::mem::zeroed() };
    let mut sslen = std::mem::size_of::<libc::sockaddr_storage>() as libc::socklen_t;
    let fd = retry(|| unsafe {
        sslen = std::mem::size_of::<libc::sockaddr_storage>() as libc::socklen_t;
        libc::accept4(sock.fd, &mut ss as *mut _ as *mut libc::sockaddr, &mut sslen, accept_flags) as isize
    })
    .map_err(map_errno)?;
    let name = net().sockets.get(&s).map(|sl| sl.name.clone()).unwrap_or_default();
    // The console's own options cross accept: the kept flags, the
    // keep-alive values and the no-push TCP flag (measured on hardware).
    let opts = {
        let listener = net().sockets.get(&s).map(|sl| sl.opts.clone()).unwrap_or_default();
        SockOptState {
            kept_flags: listener.kept_flags,
            tcp_keep_idle: listener.tcp_keep_idle,
            tcp_keep_interval: listener.tcp_keep_interval,
            tcp_keep_count: listener.tcp_keep_count,
            tcp_flags: listener.tcp_flags & 0x1000,
            // A listener never takes a segment, so it holds its family's.
            tcp_maximum_segment: listener.tcp_maximum_segment,
            ..SockOptState::default()
        }
    };
    let protocol = net().sockets.get(&s).map(|sl| sl.protocol).unwrap_or(libc::IPPROTO_TCP as i32);
    let slot = SocketSlot { transport: Arc::new(HostSocket { fd: fd as i32 }), p2p: None, name, timeouts_us: [0; 2], opts, protocol };
    let id = alloc_socket(slot).ok_or(EMFILE_)?;
    if addr != 0 || addrlen != 0 {
        let r = match (addr, addrlen) {
            (0, _) => Err(EFAULT_),
            (_, 0) => Err(EINVAL_),
            _ => convert_host_sockaddr(ctx, sockaddr_to(&ss, sslen), addr, addrlen).map(|_| ()),
        };
        if let Err(e) = r {
            net().sockets.remove(&id);
            return Err(e);
        }
    }
    Ok(id as i64)
}

fn cb_accept(ctx: &mut CpuContext) -> HleResult {
    let r = net_accept(ctx, ctx.rdi() as i32, ctx.rsi(), ctx.rdx());
    socket_result(ctx, r)
}

/// sceNetErrnoLoc() -> int*
fn cb_errno_loc(ctx: &mut CpuContext) -> HleResult {
    net_errno_address(ctx) as i64
}

/// sceNetPoolCreate(name, size, flags) -> pool id (1..31)
fn cb_pool_create(ctx: &mut CpuContext) -> HleResult {
    let size = ctx.rsi() as i32;
    if size == 0 {
        return finish(ctx, net_error(EINVAL_));
    }
    let name = read_name(ctx, ctx.rdi(), MAX_NAME_LENGTH);
    let mut st = net();
    match (1..POOLS_MAX).find(|&i| st.pools[i].is_none()) {
        Some(i) => {
            st.pools[i] = Some(NetPool { name, size });
            i as i64
        }
        None => {
            drop(st);
            finish(ctx, net_error(ENFILE_))
        }
    }
}

/// Test hook: a libSceNet TCP socket connected to 127.0.0.1:`port`.
#[cfg(test)]
pub(crate) fn tests_connected_socket(port: u16) -> i32 {
    let sock = HostSocket::new(libc::AF_INET, libc::SOCK_STREAM, 0).expect("host socket");
    let (ss, len) = sockaddr_from(&SocketAddr::from(([127, 0, 0, 1], port)));
    assert_eq!(unsafe { libc::connect(sock.fd, &ss as *const _ as *const libc::sockaddr, len) }, 0);
    alloc_socket(SocketSlot { transport: Arc::new(sock), p2p: None, name: "test".into(), timeouts_us: [0; 2], opts: SockOptState::default(), protocol: libc::IPPROTO_TCP as i32 }).expect("socket id")
}

/// Test hook: sceNetSocketClose(id) without the guest call.
#[cfg(test)]
pub(crate) fn tests_close_socket(id: i32) -> bool {
    net().sockets.remove(&id).is_some()
}

/// Test hook: sceNetPoolCreate(name, 0x4000, 0).
#[cfg(test)]
pub(crate) fn tests_pool(ctx: &mut CpuContext, name: u64) -> u64 {
    ctx.set(crate::context::CpuRegister::Rdi, name);
    ctx.set(crate::context::CpuRegister::Rsi, 0x4000);
    ctx.set(crate::context::CpuRegister::Rdx, 0);
    cb_pool_create(ctx) as u64
}

fn cb_pool_destroy(ctx: &mut CpuContext) -> HleResult {
    let id = ctx.rdi() as i32;
    let removed = {
        let mut st = net();
        if st.pool_valid(id) { st.pools[id as usize].take() } else { None }
    };
    finish(ctx, if removed.is_some() { 0 } else { net_error(EBADF_) })
}

/// sceNetGetMemoryPoolStats(memid, SceNetMemoryPoolStats* stats): poolSize,
/// maxInuseSize, currentInuseSize (size_t each). Sockets live on the host,
/// so nothing is allocated from the pool.
fn cb_get_memory_pool_stats(ctx: &mut CpuContext) -> HleResult {
    let (id, out) = (ctx.rdi() as i32, ctx.rsi());
    // The pool is checked before the output, as the console does.
    let size = {
        let st = net();
        if st.pool_valid(id) { st.pools[id as usize].as_ref().map(|p| p.size) } else { None }
    };
    let Some(size) = size else {
        return finish(ctx, net_error(EBADF_));
    };
    if out == 0 {
        return finish(ctx, net_error(EINVAL_));
    }
    let mut b = [0u8; 24];
    b[0..8].copy_from_slice(&(size as u32 as u64).to_le_bytes());
    if ctx.write_bytes(out, &b) { 0 } else { finish(ctx, net_error(EFAULT_)) }
}

fn cb_htonl(ctx: &mut CpuContext) -> HleResult {
    (ctx.rdi() as u32).swap_bytes() as i64
}

fn cb_htons(ctx: &mut CpuContext) -> HleResult {
    (ctx.rdi() as u16).swap_bytes() as i64
}

/// sceNetResolverCreate(name, memid, flags) -> resolver id (0..31)
fn cb_resolver_create(ctx: &mut CpuContext) -> HleResult {
    let (name_addr, memid, flags) = (ctx.rdi(), ctx.rsi() as i32, ctx.rdx() as i32);
    if name_addr == 0 || flags != 0 {
        return finish(ctx, net_error(EINVAL_));
    }
    let name = read_name(ctx, name_addr, MAX_NAME_LENGTH);
    let mut st = net();
    let slot = if st.pool_valid(memid) { st.resolvers.iter().position(|r| r.is_none()) } else { None };
    match slot {
        Some(i) => {
            st.resolvers[i] = Some(Resolver { name, memid, last_error: 0 });
            i as i64
        }
        None => {
            drop(st);
            finish(ctx, NET_ERROR_RESOLVER_ENOSPACE)
        }
    }
}

fn cb_resolver_destroy(ctx: &mut CpuContext) -> HleResult {
    let id = ctx.rdi() as i32;
    let found = {
        let mut st = net();
        st.resolver_mut(id).is_some() && st.resolvers[id as usize].take().is_some()
    };
    finish(ctx, if found { 0 } else { net_error(EBADF_) })
}

/// sceNetResolverAbort(rid, flags): lookups complete synchronously, so
/// there is never one in flight to abort.
fn cb_resolver_abort(ctx: &mut CpuContext) -> HleResult {
    net_log(format_args!("resolver.abort rid={} flags={:#x}", ctx.rdi() as i32, ctx.rsi() as u32));
    0
}

/// sceNetResolverGetError(rid, int* status)
fn cb_resolver_get_error(ctx: &mut CpuContext) -> HleResult {
    let id = ctx.rdi() as i32;
    let status = ctx.rsi();
    if status == 0 {
        return finish(ctx, net_error(EINVAL_));
    }
    let Some(last) = net().resolver_mut(id).map(|r| r.last_error) else {
        return finish(ctx, net_error(EBADF_));
    };
    if ctx.write_u32(status, last as u32) { 0 } else { finish(ctx, net_error(EFAULT_)) }
}

fn set_resolver_error(id: i32, error: i32) {
    if let Some(r) = net().resolver_mut(id) {
        r.last_error = error;
    }
}

/// Strict `inet_pton` text form: dotted decimal IPv4 / RFC 4291 IPv6.
fn inet_pton(af: i32, text: &[u8]) -> Option<Vec<u8>> {
    let text = std::str::from_utf8(text).ok()?;
    match af {
        ORBIS_AF_INET => text.parse::<Ipv4Addr>().ok().map(|a| a.octets().to_vec()),
        _ => text.parse::<Ipv6Addr>().ok().map(|a| a.octets().to_vec()),
    }
}

/// sceNetResolverStartNtoa(rid, hostname, SceNetInAddr* addr, timeout,
/// retry, flags): dotted IPv4 literals resolve locally; the console is
/// offline, so names fail with "no DNS".
fn cb_resolver_start_ntoa(ctx: &mut CpuContext) -> HleResult {
    let (id, host_addr, dst, flags) = (ctx.rdi() as i32, ctx.rsi(), ctx.rdx(), ctx.r9() as i32);
    let (timeout, retry) = (ctx.rcx() as i32, ctx.r8() as i32);
    // A negative timeout or retry count and any flag bit but 0x1 and
    // 0x10000 are EINVAL before the resolver is looked up.
    if host_addr == 0 || dst == 0 || timeout < 0 || retry < 0 || flags as u32 & !0x0001_0001 != 0 {
        return finish(ctx, net_error(EINVAL_));
    }
    if net().resolver_mut(id).is_none() {
        return finish(ctx, net_error(EBADF_));
    }
    let host = ctx.read_cstr_bytes(host_addr, MAX_NAME_LENGTH).unwrap_or_default();
    if flags & 0x10000 == 0 {
        if let Some(octets) = inet_pton(ORBIS_AF_INET, &host) {
            return if ctx.write_bytes(dst, &octets) { 0 } else { finish(ctx, net_error(EFAULT_)) };
        }
    }
    set_resolver_error(id, NET_ERROR_RESOLVER_NO_DNS);
    log::info!("[net] resolver: '{}' -> no DNS (offline)", String::from_utf8_lossy(&host));
    finish(ctx, NET_ERROR_RESOLVER_NO_DNS)
}

/// sceNetInetPton(af, src, dst) -> 1 ok, 0 unparseable.
/// sceNetUsleep(int usec): a cancellable sleep; a negative time is EINVAL.
fn cb_net_usleep(ctx: &mut CpuContext) -> HleResult {
    let us = ctx.rdi() as i32;
    if us < 0 {
        return net_fail(ctx, EINVAL_);
    }
    crate::kernel::sleep_micros(us as u64);
    0
}

/// sceNetInetPtonEx(af, src, dst, flags): IPv4 text is read the way
/// inet_aton reads it (1 to 4 parts, octal and hex parts, ended by
/// whitespace), as the console does; other families are sceNetInetPton.
/// The flags are not looked at.
fn cb_net_inet_pton_ex(ctx: &mut CpuContext) -> HleResult {
    let (af, src, dst) = (ctx.rdi() as i32, ctx.rsi(), ctx.rdx());
    if af != ORBIS_AF_INET {
        return cb_net_inet_pton(ctx);
    }
    if src == 0 {
        return finish(ctx, net_error(EINVAL_));
    }
    let Some(text) = ctx.read_cstr_bytes(src, MAX_NAME_LENGTH) else {
        return finish(ctx, net_error(EFAULT_));
    };
    match inet_aton(&text) {
        // A null destination still parses (measured on hardware).
        Some(_) if dst == 0 => 1,
        Some(addr) => {
            if ctx.write_bytes(dst, &addr.to_be_bytes()) {
                1
            } else {
                finish(ctx, net_error(EFAULT_))
            }
        }
        None => 0,
    }
}

/// BSD inet_aton: "a", "a.b", "a.b.c" or "a.b.c.d", each part decimal,
/// octal (leading 0) or hex (0x); the last part fills the remaining bytes.
/// The text may end at whitespace.
pub(crate) fn inet_aton(text: &[u8]) -> Option<u32> {
    let at = |i: usize| text.get(i).copied().unwrap_or(0);
    let mut parts: Vec<u32> = Vec::with_capacity(3);
    let mut t = 0;
    let mut value: u32;
    loop {
        if !at(t).is_ascii_digit() {
            return None;
        }
        let mut base = 10;
        if at(t) == b'0' {
            t += 1;
            if at(t) == b'x' || at(t) == b'X' {
                base = 16;
                t += 1;
            } else if at(t) != b'9' {
                base = 8;
            }
        }
        value = 0;
        loop {
            let c = at(t);
            if c.is_ascii_digit() {
                if base == 8 && c >= b'8' {
                    return None;
                }
                value = value.wrapping_mul(base).wrapping_add((c - b'0') as u32);
            } else if base == 16 && c.is_ascii_hexdigit() {
                value = value.wrapping_mul(16).wrapping_add((c.to_ascii_lowercase() - b'a' + 10) as u32);
            } else {
                break;
            }
            t += 1;
        }
        if at(t) != b'.' {
            break;
        }
        if parts.len() == 3 || value > 0xFF {
            return None;
        }
        parts.push(value);
        t += 1;
    }
    if at(t) != 0 && !b" \t\n\x0b\x0c\r".contains(&at(t)) {
        return None;
    }
    if value > (u32::MAX >> (8 * parts.len())) {
        return None;
    }
    Some(parts.iter().enumerate().fold(value, |a, (i, &p)| a | p << (24 - 8 * i)))
}

fn cb_net_inet_pton(ctx: &mut CpuContext) -> HleResult {
    let (af, src, dst) = (ctx.rdi() as i32, ctx.rsi(), ctx.rdx());
    if src == 0 {
        return finish(ctx, net_error(EINVAL_));
    }
    if af != ORBIS_AF_INET && af != ORBIS_AF_INET6 {
        return finish(ctx, net_error(EAFNOSUPPORT_));
    }
    // A null destination is refused for IPv6 only; IPv4 still checks the
    // text (measured on hardware).
    if dst == 0 && af == ORBIS_AF_INET6 {
        return finish(ctx, net_error(EINVAL_));
    }
    let Some(text) = ctx.read_cstr_bytes(src, MAX_NAME_LENGTH) else {
        return finish(ctx, net_error(EFAULT_));
    };
    // The console's IPv4 parser stops at whitespace; an IPv6 text is read
    // whole.
    let head: Vec<u8> = if af == ORBIS_AF_INET {
        let end = text.iter().position(|c| b" \t\n\x0b\x0c\r".contains(c)).unwrap_or(text.len());
        text[..end].to_vec()
    } else {
        text
    };
    // An IPv6 text ending in a dotted IPv4 tail has that tail checked
    // first: an unparseable one fails without an IPv6 attempt.
    if af == ORBIS_AF_INET6 {
        let whole = String::from_utf8_lossy(&head).into_owned();
        if let Some(colon) = whole.rfind(':') {
            if whole[colon..].contains('.') && whole[colon + 1..].parse::<Ipv4Addr>().is_err() {
                return 0;
            }
        }
    }
    match inet_pton(af, &head) {
        Some(bytes) => {
            if dst != 0 && !ctx.write_bytes(dst, &bytes) {
                return finish(ctx, net_error(EFAULT_));
            }
            1
        }
        None => 0,
    }
}

// ---------------------------------------------------------------------------
// libSceNet: socket queries / shutdown / epoll
// ---------------------------------------------------------------------------

const SO_SNDTIMEO: i32 = 0x1005;
const SO_RCVTIMEO: i32 = 0x1006;
const SO_TYPE: i32 = 0x1008;
const SO_PROTOCOL: i32 = 0x1016;

const NET_EPOLL_CTL_ADD: i32 = 1;
const NET_EPOLL_CTL_MOD: i32 = 2;
const NET_EPOLL_CTL_DEL: i32 = 3;
const NET_EPOLLIN: u32 = 0x01;
const NET_EPOLLOUT: u32 = 0x02;
const NET_EPOLLERR: u32 = 0x08;
/// SceNetEpollEvent: events, reserved, ident (u64), data (u64).
const NET_EPOLL_EVENT_SIZE: u64 = 24;

/// SceNetSockInfo entry size (name[32], pid, s, type, policy, queues,
/// addresses, ports, state, flags, reserved[8]).
const NET_SOCK_INFO_SIZE: usize = 0xA0;
const NET_SOCKINFO_F_SELF: i32 = 0x1;
const NET_SOCKINFO_F_NONBLOCK: i32 = 0x20_0000;
/// The one sceNetGetSockInfo flag: also list sockets in TIME_WAIT.
const NET_SOCKINFO_TIME_WAIT: i32 = 0x40;

/// sceNetGetsockname(s, addr, addrlen*): UDPP2P sockets report their
/// virtual port after the address.
fn net_getsockname(ctx: &CpuContext, s: i32, addr: u64, len_addr: u64) -> Result<(), i32> {
    let slot = socket_slot(s)?;
    if addr == 0 || len_addr == 0 {
        return Err(EINVAL_);
    }
    let local = local_addr(slot.transport.fd).ok_or_else(|| map_errno(last_errno()))?;
    let copied = convert_host_sockaddr(ctx, Some(local), addr, len_addr)?;
    if let Some(p2p) = slot.p2p.filter(|_| copied >= 10) {
        if !ctx.write_bytes(addr + 8, &p2p.virtual_port) {
            return Err(EFAULT_);
        }
    }
    Ok(())
}

fn cb_getsockname(ctx: &mut CpuContext) -> HleResult {
    let r = net_getsockname(ctx, ctx.rdi() as i32, ctx.rsi(), ctx.rdx());
    socket_result(ctx, r.map(|_| 0))
}

fn peer_addr(fd: i32) -> Option<SocketAddr> {
    let mut ss: libc::sockaddr_storage = unsafe { std::mem::zeroed() };
    let mut len = std::mem::size_of::<libc::sockaddr_storage>() as libc::socklen_t;
    let r = unsafe { libc::getpeername(fd, &mut ss as *mut _ as *mut libc::sockaddr, &mut len) };
    if r < 0 {
        None
    } else {
        sockaddr_to(&ss, len)
    }
}

/// One SceNetSockInfo record for socket `id`.
fn sock_info_record(id: i32, name: &str, sock: &HostSocket) -> [u8; NET_SOCK_INFO_SIZE] {
    let mut b = [0u8; NET_SOCK_INFO_SIZE];
    let n = name.len().min(31);
    b[..n].copy_from_slice(&name.as_bytes()[..n]);
    b[32..36].copy_from_slice(&(std::process::id() as i32).to_le_bytes());
    b[36..40].copy_from_slice(&id.to_le_bytes());
    let stream = get_int_opt(sock.fd, libc::SOL_SOCKET, libc::SO_TYPE) == Ok(libc::SOCK_STREAM);
    b[40] = if stream { 1 } else { 2 };
    let mut queued: i32 = 0;
    if unsafe { libc::ioctl(sock.fd, libc::FIONREAD, &mut queued) } == 0 {
        b[44..48].copy_from_slice(&queued.to_le_bytes());
    }
    let mut pending: i32 = 0;
    if unsafe { libc::ioctl(sock.fd, libc::TIOCOUTQ, &mut pending) } == 0 {
        b[48..52].copy_from_slice(&pending.to_le_bytes());
    }
    let local = local_addr(sock.fd);
    let peer = peer_addr(sock.fd);
    if let Some(l) = local {
        b[52..56].copy_from_slice(&map_to_ipv4(l.ip()));
        b[60..62].copy_from_slice(&l.port().to_be_bytes());
    }
    if let Some(p) = peer {
        b[56..60].copy_from_slice(&map_to_ipv4(p.ip()));
        b[62..64].copy_from_slice(&p.port().to_be_bytes());
    }
    // state: 1 = connected / bound, 0 = idle.
    let state: i32 = if peer.is_some() { 1 } else { 0 };
    b[68..72].copy_from_slice(&state.to_le_bytes());
    let flags = NET_SOCKINFO_F_SELF | if sock.blocking() { 0 } else { NET_SOCKINFO_F_NONBLOCK };
    b[72..76].copy_from_slice(&flags.to_le_bytes());
    let rcvbuf = get_int_opt(sock.fd, libc::SOL_SOCKET, libc::SO_RCVBUF).unwrap_or(0);
    let sndbuf = get_int_opt(sock.fd, libc::SOL_SOCKET, libc::SO_SNDBUF).unwrap_or(0);
    b[0x64..0x68].copy_from_slice(&rcvbuf.to_le_bytes());
    b[0x68..0x6C].copy_from_slice(&sndbuf.to_le_bytes());
    b
}

fn is_ipv6_socket(sock: &HostSocket) -> bool {
    get_int_opt(sock.fd, libc::SOL_SOCKET, libc::SO_DOMAIN) == Ok(libc::AF_INET6)
}

/// sceNetGetSockInfo(s, info, n, flags): `s` < 0 reports every IPv4 socket,
/// up to `n` records (with no buffer: how many there are). Flags other than
/// TIME_WAIT are EINVAL; an IPv6 socket asked for by number EAFNOSUPPORT.
/// Returns the number of records written.
fn cb_get_sock_info(ctx: &mut CpuContext) -> HleResult {
    let id = ctx.rdi() as i32;
    let info = ctx.rsi();
    let count = ctx.rdx() as i32;
    let flags = ctx.rcx() as i32;
    if flags & !NET_SOCKINFO_TIME_WAIT != 0 {
        return net_fail(ctx, EINVAL_);
    }
    let listed = |st: &NetState| -> Vec<i32> {
        let mut ids: Vec<i32> = st.sockets.iter().filter(|(_, s)| !is_ipv6_socket(&s.transport)).map(|(&i, _)| i).collect();
        ids.sort_unstable();
        ids
    };
    if info == 0 {
        return if id < 0 { listed(&net()).len() as i64 } else { net_fail(ctx, EINVAL_) };
    }
    if count < 1 {
        return net_fail(ctx, EINVAL_);
    }
    let selected: Vec<(i32, SocketSlot)> = {
        let st = net();
        let ids: Vec<i32> = if id < 0 {
            listed(&st)
        } else if let Some(s) = st.sockets.get(&id) {
            if is_ipv6_socket(&s.transport) {
                drop(st);
                return net_fail(ctx, EAFNOSUPPORT_);
            }
            vec![id]
        } else {
            drop(st);
            return net_fail(ctx, EBADF_);
        };
        ids.into_iter().take(count as usize).map(|i| (i, st.sockets[&i].clone())).collect()
    };
    for (i, (sid, slot)) in selected.iter().enumerate() {
        let rec = sock_info_record(*sid, &slot.name, &slot.transport);
        if !ctx.write_bytes(info + (i * NET_SOCK_INFO_SIZE) as u64, &rec) {
            return net_fail(ctx, EFAULT_);
        }
    }
    selected.len() as i64
}

/// sceNetShutdown(s, how)
fn cb_shutdown(ctx: &mut CpuContext) -> HleResult {
    let how = ctx.rsi() as i32;
    let r = native_socket(ctx.rdi() as i32).and_then(|sock| {
        if !(0..=2).contains(&how) {
            return Err(EINVAL_);
        }
        int_result(unsafe { libc::shutdown(sock.fd, how) })
    });
    NET_CHANGED.notify_all();
    socket_result(ctx, r.map(|_| 0))
}

/// sceNetSocketAbort(s, flags): wakes every thread blocked on the socket by
/// shutting both directions down.
fn cb_socket_abort(ctx: &mut CpuContext) -> HleResult {
    let id = ctx.rdi() as i32;
    let slot = match socket_slot(id) {
        Ok(s) => s,
        Err(e) => return net_fail(ctx, e),
    };
    unsafe { libc::shutdown(slot.transport.fd, libc::SHUT_RDWR) };
    NET_CHANGED.notify_all();
    net_log(format_args!("socket.abort id={id}"));
    0
}

/// sceNetEpollCreate(name, flags) -> epoll id (1024..1055)
fn cb_epoll_create(ctx: &mut CpuContext) -> HleResult {
    let (name_addr, flags) = (ctx.rdi(), ctx.rsi() as i32);
    if name_addr == 0 || flags != 0 {
        return net_fail(ctx, EINVAL_);
    }
    let name = read_name(ctx, name_addr, 32);
    let mut st = net();
    match st.epolls.iter().position(|e| e.is_none()) {
        Some(i) => {
            st.epolls[i] = Some(NetEpoll { name, registrations: Vec::new(), aborted: false });
            (EPOLL_ID_BASE + i as i32) as i64
        }
        None => {
            drop(st);
            net_fail(ctx, EMFILE_)
        }
    }
}

/// sceNetEpollDestroy(eid)
fn cb_epoll_destroy(ctx: &mut CpuContext) -> HleResult {
    let id = ctx.rdi() as i32;
    let found = {
        let mut st = net();
        match st.epoll_mut(id) {
            Some(_) => {
                st.epolls[(id - EPOLL_ID_BASE) as usize] = None;
                true
            }
            None => false,
        }
    };
    if !found {
        return net_fail(ctx, EBADF_);
    }
    NET_CHANGED.notify_all();
    0
}

/// sceNetEpollAbort(eid, flags): wakes a wait on the epoll, or makes the
/// next one fail, with ECONNABORTED.
fn cb_epoll_abort(ctx: &mut CpuContext) -> HleResult {
    let id = ctx.rdi() as i32;
    let found = match net().epoll_mut(id) {
        Some(ep) => {
            ep.aborted = true;
            true
        }
        None => false,
    };
    if !found {
        return net_fail(ctx, EBADF_);
    }
    NET_CHANGED.notify_all();
    0
}

/// The state of epoll `eid` at a wait step: gone (EBADF), aborted
/// (ECONNABORTED, consuming the abort) or still waitable (0).
fn epoll_wait_errno(st: &mut NetState, eid: i32) -> i32 {
    match st.epoll_mut(eid) {
        None => EBADF_,
        Some(ep) if ep.aborted => {
            ep.aborted = false;
            ECONNABORTED_
        }
        Some(_) => 0,
    }
}

/// sceNetEpollControl(eid, op, id, const SceNetEpollEvent* event)
fn cb_epoll_control(ctx: &mut CpuContext) -> HleResult {
    let eid = ctx.rdi() as i32;
    let op = ctx.rsi() as i32;
    let sid = ctx.rdx() as i32;
    let event = ctx.rcx();
    if (op == NET_EPOLL_CTL_ADD || op == NET_EPOLL_CTL_MOD) && event == 0 {
        return net_fail(ctx, EINVAL_);
    }
    if op == NET_EPOLL_CTL_DEL && event != 0 {
        return net_fail(ctx, EINVAL_);
    }
    if !(NET_EPOLL_CTL_ADD..=NET_EPOLL_CTL_DEL).contains(&op) {
        return net_fail(ctx, EINVAL_);
    }
    let requested = if op == NET_EPOLL_CTL_DEL {
        None
    } else {
        match (ctx.read_u32(event), ctx.read_u64(event + 16)) {
            (Some(ev), Some(data)) => Some((ev, data)),
            _ => return net_fail(ctx, EFAULT_),
        }
    };
    let errno = {
        let mut st = net();
        if !st.sockets.contains_key(&sid) {
            EBADF_
        } else {
            match st.epoll_mut(eid) {
                None => EBADF_,
                Some(ep) => {
                    let pos = ep.registrations.iter().position(|r| r.0 == sid);
                    match (op, pos, requested) {
                        (NET_EPOLL_CTL_ADD, Some(_), _) => EEXIST_,
                        (NET_EPOLL_CTL_ADD, None, Some((ev, data))) => {
                            ep.registrations.push((sid, ev, data));
                            0
                        }
                        (NET_EPOLL_CTL_MOD, Some(i), Some((ev, data))) => {
                            ep.registrations[i] = (sid, ev, data);
                            0
                        }
                        (NET_EPOLL_CTL_DEL, Some(i), _) => {
                            ep.registrations.remove(i);
                            0
                        }
                        _ => ENOENT_,
                    }
                }
            }
        }
    };
    if errno != 0 {
        return net_fail(ctx, errno);
    }
    NET_CHANGED.notify_all();
    0
}

/// Poll slice while waiting, so registration changes, destruction and guest
/// signals are noticed promptly.
const EPOLL_SLICE: std::time::Duration = std::time::Duration::from_millis(10);

/// Readiness of one registration in select() terms: readable on data,
/// end-of-stream or error; writable on space or error; exceptional on
/// out-of-band data.
fn epoll_ready(requested: u32, revents: i16) -> u32 {
    let mut ready = 0;
    if requested & NET_EPOLLIN != 0 && revents & (libc::POLLIN | libc::POLLHUP | libc::POLLERR) != 0 {
        ready |= NET_EPOLLIN;
    }
    if requested & NET_EPOLLOUT != 0 && revents & (libc::POLLOUT | libc::POLLERR) != 0 {
        ready |= NET_EPOLLOUT;
    }
    if revents & libc::POLLPRI != 0 {
        ready |= NET_EPOLLERR;
    }
    ready
}

/// sceNetEpollWait(eid, events, maxevents, timeout_us): -1 waits forever,
/// 0 polls. Returns the number of ready sockets.
fn cb_epoll_wait(ctx: &mut CpuContext) -> HleResult {
    let eid = ctx.rdi() as i32;
    let events_addr = ctx.rsi();
    let max_events = ctx.rdx() as i32;
    let timeout = ctx.rcx() as i32;
    if events_addr == 0 {
        return net_fail(ctx, EFAULT_);
    }
    if max_events <= 0 || timeout < -1 {
        return net_fail(ctx, EINVAL_);
    }
    let deadline = (timeout >= 0).then(|| std::time::Instant::now() + std::time::Duration::from_micros(timeout as u64));
    // An epoll without registrations waits for one to be added.
    {
        let mut st = net();
        loop {
            let errno = epoll_wait_errno(&mut st, eid);
            if errno != 0 {
                drop(st);
                return net_fail(ctx, errno);
            }
            if st.epoll_mut(eid).is_some_and(|ep| !ep.registrations.is_empty()) {
                break;
            }
            if timeout == 0 {
                return 0;
            }
            let (g, timed_out) = crate::signals::wait_cv(&*NET, &NET_CHANGED, st, deadline);
            st = g;
            if timed_out {
                let errno = epoll_wait_errno(&mut st, eid);
                drop(st);
                return if errno != 0 { net_fail(ctx, errno) } else { 0 };
            }
        }
    }
    loop {
        let regs: Vec<(i32, u32, u64, SocketSlot)> = {
            let mut st = net();
            let errno = epoll_wait_errno(&mut st, eid);
            if errno != 0 {
                drop(st);
                return net_fail(ctx, errno);
            }
            let Some(ep) = st.epoll_mut(eid) else {
                drop(st);
                return net_fail(ctx, EBADF_);
            };
            let regs = ep.registrations.clone();
            regs.into_iter().filter_map(|(sid, ev, data)| st.sockets.get(&sid).map(|s| (sid, ev, data, s.clone()))).collect()
        };
        if regs.is_empty() {
            return 0;
        }
        let mut fds: Vec<libc::pollfd> = regs
            .iter()
            .map(|(_, ev, _, s)| {
                let mut events = libc::POLLPRI;
                if ev & NET_EPOLLIN != 0 {
                    events |= libc::POLLIN;
                }
                if ev & NET_EPOLLOUT != 0 {
                    events |= libc::POLLOUT;
                }
                libc::pollfd { fd: s.transport.fd, events, revents: 0 }
            })
            .collect();
        let slice = match deadline {
            None => EPOLL_SLICE,
            Some(d) => d.saturating_duration_since(std::time::Instant::now()).min(EPOLL_SLICE),
        };
        let wait_ms = slice.as_micros().div_ceil(1000) as i32;
        let ready = unsafe { libc::poll(fds.as_mut_ptr(), fds.len() as libc::nfds_t, wait_ms) };
        if ready < 0 {
            let e = last_errno();
            if e != libc::EINTR {
                return net_fail(ctx, map_errno(e));
            }
        }
        // Destroyed or aborted while polling.
        let errno = epoll_wait_errno(&mut net(), eid);
        if errno != 0 {
            return net_fail(ctx, errno);
        }
        let mut written = 0usize;
        for (pfd, (sid, ev, data, slot)) in fds.iter().zip(&regs) {
            let ready = epoll_ready(*ev, pfd.revents);
            if ready == 0 {
                continue;
            }
            if ready & NET_EPOLLIN != 0 && slot.p2p.is_some() {
                return net_fail(ctx, EOPNOTSUPP_);
            }
            let mut rec = [0u8; NET_EPOLL_EVENT_SIZE as usize];
            rec[0..4].copy_from_slice(&ready.to_le_bytes());
            rec[8..16].copy_from_slice(&(*sid as u64).to_le_bytes());
            rec[16..24].copy_from_slice(&data.to_le_bytes());
            if !ctx.write_bytes(events_addr + written as u64 * NET_EPOLL_EVENT_SIZE, &rec) {
                return net_fail(ctx, EFAULT_);
            }
            written += 1;
            if written == max_events as usize {
                break;
            }
        }
        if written > 0 {
            return written as i64;
        }
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return 0;
        }
        if crate::signals::pending_for_current() {
            crate::signals::dispatch_pending();
        }
    }
}

/// sceNetResolverStartAton(rid, const SceNetInAddr* addr, char* hostname,
/// int len, timeout, retry, flags): reverse lookups need a DNS server,
/// which does not exist offline.
fn cb_resolver_start_aton(ctx: &mut CpuContext) -> HleResult {
    let id = ctx.rdi() as i32;
    let (addr, hostname, len) = (ctx.rsi(), ctx.rdx(), ctx.rcx() as i32);
    if addr == 0 || hostname == 0 || len <= 0 {
        return finish(ctx, net_error(EINVAL_));
    }
    if net().resolver_mut(id).is_none() {
        return finish(ctx, net_error(EBADF_));
    }
    let Some(raw) = ctx.read_bytes(addr, 4) else {
        return finish(ctx, net_error(EFAULT_));
    };
    set_resolver_error(id, NET_ERROR_RESOLVER_NO_DNS);
    log::info!("[net] resolver: reverse {} -> no DNS (offline)", Ipv4Addr::new(raw[0], raw[1], raw[2], raw[3]));
    finish(ctx, NET_ERROR_RESOLVER_NO_DNS)
}

/// SceNetResolverInfo: 10 records of { 16-byte address, i32 af at +16,
/// 12 reserved bytes }, then the record count (+320), the IPv4 record count
/// (+324) and 56 reserved bytes.
const RESOLVER_RECORD_SIZE: usize = 32;
const RESOLVER_RECORDS: usize = 10;
const RESOLVER_INFO_COUNT: usize = RESOLVER_RECORDS * RESOLVER_RECORD_SIZE;
const RESOLVER_INFO_SIZE: usize = RESOLVER_INFO_COUNT + 8 + 14 * 4;

/// sceNetResolverStartNtoaMultipleRecords(rid, hostname, SceNetResolverInfo*
/// info, timeout, retry, flags): address literals resolve locally; names
/// fail with "no DNS". The info is only written on success.
fn cb_resolver_start_ntoa_multiple(ctx: &mut CpuContext) -> HleResult {
    let (id, host_addr, info) = (ctx.rdi() as i32, ctx.rsi(), ctx.rdx());
    if host_addr == 0 || info == 0 {
        return finish(ctx, net_error(EINVAL_));
    }
    if net().resolver_mut(id).is_none() {
        return finish(ctx, net_error(EBADF_));
    }
    let host = ctx.read_cstr_bytes(host_addr, MAX_NAME_LENGTH).unwrap_or_default();
    let literal = inet_pton(ORBIS_AF_INET, &host)
        .map(|b| (ORBIS_AF_INET, b))
        .or_else(|| inet_pton(ORBIS_AF_INET6, &host).map(|b| (ORBIS_AF_INET6, b)));
    if let Some((af, bytes)) = literal {
        let mut b = [0u8; RESOLVER_INFO_SIZE];
        b[..bytes.len()].copy_from_slice(&bytes);
        b[16..20].copy_from_slice(&af.to_le_bytes());
        b[RESOLVER_INFO_COUNT..RESOLVER_INFO_COUNT + 4].copy_from_slice(&1u32.to_le_bytes());
        b[RESOLVER_INFO_COUNT + 4..RESOLVER_INFO_COUNT + 8].copy_from_slice(&((af == ORBIS_AF_INET) as u32).to_le_bytes());
        if !ctx.write_bytes(info, &b) {
            return finish(ctx, net_error(EFAULT_));
        }
        set_resolver_error(id, 0);
        return 0;
    }
    set_resolver_error(id, NET_ERROR_RESOLVER_NO_DNS);
    log::info!("[net] resolver: '{}' (multiple records) -> no DNS (offline)", String::from_utf8_lossy(&host));
    finish(ctx, NET_ERROR_RESOLVER_NO_DNS)
}

// ---------------------------------------------------------------------------
// libkernel sockets
// ---------------------------------------------------------------------------

const SOCK_STREAM_K: i32 = 1;
const SOCK_DGRAM_K: i32 = 2;
const SOCK_CLOEXEC_K: i32 = 0x1000_0000;
const SOCK_NONBLOCK_K: i32 = 0x2000_0000;
const AF_UNIX_K: i32 = 1;
const EPERM_: i32 = 1;
const EPROTOTYPE_: i32 = 41;
const ESOCKTNOSUPPORT_: i32 = 44;
const EPFNOSUPPORT_: i32 = 46;
const ETOOMANYREFS_: i32 = 59;
const EHOSTDOWN_: i32 = 64;
const EPROTO_: i32 = 92;

const SO_DONTROUTE: i32 = 0x0010;
const SO_LINGER: i32 = 0x0080;
const SO_OOBINLINE: i32 = 0x0100;
const SO_ACCEPTCONN: i32 = 0x0002;
const SO_NOSIGPIPE: i32 = 0x0800;
/// TCP_NODELAY as FreeBSD reports it (the TF_NODELAY bit of t_flags).
const TF_NODELAY: i32 = 4;
/// Longest SO_LINGER time FreeBSD accepts.
const MAX_LINGER: i32 = 65535;

/// The SOL_SOCKET options that are plain flags: guest option -> host
/// option. getsockopt reports an enabled flag as its own bit, as FreeBSD
/// returns `so_options & option`. SO_REUSEADDR and SO_REUSEPORT are kept
/// apart (see [`KEPT_FLAG_OPTIONS`]).
const FLAG_OPTIONS: [(i32, i32); 4] = [
    (SO_KEEPALIVE, libc::SO_KEEPALIVE),
    (SO_DONTROUTE, libc::SO_DONTROUTE),
    (SO_BROADCAST, libc::SO_BROADCAST),
    (SO_OOBINLINE, libc::SO_OOBINLINE),
];

/// The SOL_SOCKET flag bits the console keeps in its own state (measured
/// on hardware): SO_REUSEADDR and SO_REUSEPORT also reach the host socket,
/// the rest are stored only.
const KEPT_FLAG_OPTIONS: i32 = 0x1 | SO_REUSEADDR | 0x40 | SO_REUSEPORT | 0x400 | 0x2000 | 0x10000;
/// The SOL_SOCKET flag bits a setsockopt accepts and never reports back.
const WRITE_ONLY_FLAG_OPTIONS: i32 = 0x4000 | 0x8000;
/// The smallest TCP maximum segment the console accepts.
const TCP_MINIMUM_SEGMENT: i32 = 216;
/// TCP keep-alive values are ticks of a millisecond clock.
const TCP_TICKS_PER_SECOND: u32 = 1000;
const TCP_DEFAULT_KEEP_IDLE: u32 = 7200;
const TCP_DEFAULT_KEEP_INTERVAL: u32 = 75;
const TCP_DEFAULT_KEEP_COUNT: u32 = 8;

/// The socket options the console keeps in its own state apart from the
/// host stack (the values and their validations measured on hardware).
#[derive(Clone, Debug)]
struct SockOptState {
    /// The kept SOL_SOCKET flag bits.
    kept_flags: i32,
    /// TCP_MAXSEG as last accepted.
    tcp_maximum_segment: i32,
    /// TCP_KEEPIDLE / TCP_KEEPINTVL / TCP_KEEPCNT, in ticks.
    tcp_keep_idle: u32,
    tcp_keep_interval: u32,
    tcp_keep_count: u32,
    /// The TCP flag options: NOPUSH 0x1000, NOOPT 0x8, the option 0x80
    /// 0x10000.
    tcp_flags: i32,
    ip_time_to_live: i32,
    ip_type_of_service: i32,
    multicast_time_to_live: i32,
    multicast_loop: bool,
    multicast_interface: u32,
    /// Joined multicast groups as (group, interface) of the request.
    memberships: Vec<u64>,
    /// IP_RECVDSTADDR (1), IP_RECVIF (2), IP_RECVTTL (4).
    ip_receive_bits: i32,
}

impl SockOptState {
    /// The state of a new socket of `family`: TCP_MAXSEG starts at 1220
    /// for AF_INET6 and 536 otherwise (measured on a PS5, firmware 13.60).
    fn for_family(family: i32) -> Self {
        Self { tcp_maximum_segment: if family == ORBIS_AF_INET6 { 1220 } else { 536 }, ..Self::default() }
    }
}

impl Default for SockOptState {
    fn default() -> Self {
        Self {
            kept_flags: 0,
            tcp_maximum_segment: 536,
            tcp_keep_idle: 0,
            tcp_keep_interval: 0,
            tcp_keep_count: 0,
            tcp_flags: 0,
            ip_time_to_live: 64,
            ip_type_of_service: 0,
            multicast_time_to_live: 1,
            multicast_loop: true,
            multicast_interface: 0,
            memberships: Vec::new(),
            ip_receive_bits: 0,
        }
    }
}

/// The bit a TCP flag option uses in the stored TCP flags.
fn tcp_flag_bit(option: i32) -> i32 {
    match option {
        TCP_NOPUSH => 0x1000,
        TCP_NOOPT => 0x8,
        TCP_FLAG_0X80 => 0x10000,
        _ => 0,
    }
}

/// Whether `option` is a TCP option of the console's set.
fn tcp_option_known(option: i32) -> bool {
    matches!(option, TCP_NODELAY | TCP_MAXSEG | TCP_KEEPIDLE | TCP_KEEPINTVL | TCP_KEEPCNT) || tcp_flag_bit(option) != 0
}

/// The bit an IP receive option uses in the stored receive bits.
fn ip_receive_bit(option: i32) -> i32 {
    match option {
        IP_RECVDSTADDR => 1,
        IP_RECVIF => 2,
        IP_RECVTTL => 4,
        _ => 0,
    }
}

/// Whether `value` has exactly one bit set.
fn single_bit(value: i32) -> bool {
    value != 0 && value & (value - 1) == 0
}

/// Whether `option` is one of the kept or write-only SOL_SOCKET flags.
fn kept_or_write_only(option: i32) -> bool {
    option & (KEPT_FLAG_OPTIONS | WRITE_ONLY_FLAG_OPTIONS) != 0 && single_bit(option)
}

/// Whether `option` is one of the IP multicast options.
fn is_multicast_option(option: i32) -> bool {
    matches!(option, IP_MULTICAST_IF | IP_MULTICAST_TTL | IP_MULTICAST_LOOP | IP_ADD_MEMBERSHIP | IP_DROP_MEMBERSHIP)
}

/// Set a kept or write-only SOL_SOCKET flag: the value is stored (a
/// write-only one is dropped) and REUSEADDR / REUSEPORT also reach the
/// host socket, each from the combined state.
fn set_kept_flag_option(fd: i32, state: &mut SockOptState, option: i32, number: i32) -> Result<(), i32> {
    if option & KEPT_FLAG_OPTIONS == 0 {
        return Ok(());
    }
    let flags = if number != 0 { state.kept_flags | option } else { state.kept_flags & !option };
    if option == SO_REUSEADDR || option == SO_REUSEPORT {
        set_int_opt(fd, libc::SOL_SOCKET, libc::SO_REUSEPORT, (flags & SO_REUSEPORT != 0) as i32).map_err(map_errno)?;
        set_int_opt(fd, libc::SOL_SOCKET, libc::SO_REUSEADDR, (flags & SO_REUSEADDR != 0) as i32).map_err(map_errno)?;
    }
    state.kept_flags = flags;
    Ok(())
}

/// Store a TCP keep-alive value in ticks.
fn set_tcp_ticks(slot: &mut u32, number: i32) -> Result<(), i32> {
    if number as u32 > u32::MAX / TCP_TICKS_PER_SECOND {
        return Err(EINVAL_);
    }
    *slot = number as u32;
    Ok(())
}

/// A TCP option the console keeps in its state (TCP_NODELAY goes to the
/// host): the maximum segment, the keep-alive values and the flag bits.
fn set_tcp_state_option(fd: i32, state: &mut SockOptState, option: i32, number: i32) -> Result<(), i32> {
    let bit = tcp_flag_bit(option);
    if bit != 0 {
        state.tcp_flags = if number != 0 { state.tcp_flags | bit } else { state.tcp_flags & !bit };
        return Ok(());
    }
    match option {
        // Only a connected socket takes a segment, at most the current one
        // and at least 176; one below 216 is raised to 216 (measured on a
        // PS5, firmware 13.60).
        TCP_MAXSEG => {
            if !has_peer(fd) || number <= 0 || number > state.tcp_maximum_segment || number + 40 < TCP_MINIMUM_SEGMENT {
                return Err(EINVAL_);
            }
            state.tcp_maximum_segment = number.max(TCP_MINIMUM_SEGMENT);
            Ok(())
        }
        TCP_KEEPCNT => {
            state.tcp_keep_count = number as u32;
            Ok(())
        }
        TCP_KEEPIDLE => set_tcp_ticks(&mut state.tcp_keep_idle, number),
        TCP_KEEPINTVL => set_tcp_ticks(&mut state.tcp_keep_interval, number),
        // TCP_NODELAY, the only one the host stack sees.
        _ => set_int_opt(fd, libc::IPPROTO_TCP, libc::TCP_NODELAY, (number != 0) as i32).map_err(map_errno),
    }
}

/// Whether the host socket has a peer (a connected stream socket).
fn has_peer(fd: i32) -> bool {
    let mut ss: libc::sockaddr_storage = unsafe { std::mem::zeroed() };
    let mut len = std::mem::size_of::<libc::sockaddr_storage>() as libc::socklen_t;
    // SAFETY: the address buffer and its length are valid for the call.
    unsafe { libc::getpeername(fd, &mut ss as *mut _ as *mut libc::sockaddr, &mut len) == 0 }
}

/// The value of a TCP option: the stored state, with the keep-alive
/// defaults for unset values and the console's TF_NODELAY for nodelay.
fn get_tcp_state_option(fd: i32, state: &SockOptState, option: i32) -> Result<i32, i32> {
    match option {
        TCP_MAXSEG => Ok(state.tcp_maximum_segment),
        TCP_KEEPIDLE => Ok(if state.tcp_keep_idle != 0 { state.tcp_keep_idle as i32 } else { TCP_DEFAULT_KEEP_IDLE as i32 }),
        TCP_KEEPINTVL => Ok(if state.tcp_keep_interval != 0 { state.tcp_keep_interval as i32 } else { TCP_DEFAULT_KEEP_INTERVAL as i32 }),
        TCP_KEEPCNT => Ok(if state.tcp_keep_count != 0 { state.tcp_keep_count as i32 } else { TCP_DEFAULT_KEEP_COUNT as i32 }),
        TCP_NODELAY => Ok(if get_int_opt(fd, libc::IPPROTO_TCP, libc::TCP_NODELAY).map_err(map_errno)? != 0 { TF_NODELAY } else { 0 }),
        _ => {
            let bit = tcp_flag_bit(option);
            if bit != 0 { Ok(state.tcp_flags & bit) } else { Err(ENOPROTOOPT_) }
        }
    }
}

/// An IP-level option: memberships (joined through the host stack, with
/// the console's duplicate and unknown-group checks), the multicast
/// settings, TTL / TOS and the receive bits. Multicast options need a
/// datagram socket.
fn set_ip_option(ctx: &CpuContext, fd: i32, state: &mut SockOptState, ipv4: bool, datagram: bool, option: i32, value: u64, len: u32) -> Result<(), i32> {
    if !ipv4 {
        return Err(EINVAL_);
    }
    if is_multicast_option(option) && !datagram {
        return Err(EOPNOTSUPP_);
    }
    match option {
        IP_ADD_MEMBERSHIP | IP_DROP_MEMBERSHIP => {
            if len < 8 || value == 0 {
                return Err(EINVAL_);
            }
            let b = ctx.read_bytes(value, 8).ok_or(EFAULT_)?;
            let group = u32::from_le_bytes(b[0..4].try_into().unwrap());
            let interface = u32::from_le_bytes(b[4..8].try_into().unwrap());
            // The group must be a class-D address.
            if group & 0xF0 != 0xE0 {
                return Err(EINVAL_);
            }
            let key = (u64::from(group) << 32) | u64::from(interface);
            let join = option == IP_ADD_MEMBERSHIP;
            // A drop without an interface address leaves any interface of
            // that group.
            let found = if join || interface != 0 {
                state.memberships.iter().position(|&m| m == key)
            } else {
                state.memberships.iter().position(|&m| (m >> 32) as u32 == group)
            };
            let at = match found {
                Some(at) if !join => at,
                None if !join => return Err(EADDRNOTAVAIL_),
                // A join needs no existing entry (a duplicate left above).
                _ => usize::MAX,
            };
            if join && found.is_some() {
                return Err(EADDRINUSE_);
            }
            let joined_interface = if join { interface } else { state.memberships[at] as u32 };
            let request = libc::ip_mreq {
                imr_multiaddr: libc::in_addr { s_addr: group },
                imr_interface: libc::in_addr { s_addr: joined_interface },
            };
            let native = if join { libc::IP_ADD_MEMBERSHIP } else { libc::IP_DROP_MEMBERSHIP };
            // SAFETY: one ip_mreq of its own size.
            let r = unsafe { libc::setsockopt(fd, libc::IPPROTO_IP, native, &request as *const _ as *const libc::c_void, std::mem::size_of::<libc::ip_mreq>() as libc::socklen_t) };
            if r < 0 {
                return Err(map_errno(last_errno()));
            }
            if join {
                state.memberships.push(key);
            } else {
                state.memberships.remove(at);
            }
            Ok(())
        }
        IP_MULTICAST_IF => {
            if len < 4 || value == 0 {
                return Err(EINVAL_);
            }
            let b = ctx.read_bytes(value, len.min(12) as usize).ok_or(EFAULT_)?;
            if len == 12 {
                // An ip_mreqn: only its plain address form is supported.
                let index = i32::from_le_bytes(b[8..12].try_into().unwrap());
                if index != 0 || b[4..8].iter().any(|&x| x != 0) {
                    log::warn!("setsockopt IP_MULTICAST_IF: an ip_mreqn with an interface index is not supported");
                    return Err(EINVAL_);
                }
            }
            let address = u32::from_le_bytes(b[0..4].try_into().unwrap());
            let request = libc::in_addr { s_addr: address };
            // SAFETY: one in_addr of its own size.
            let r = unsafe { libc::setsockopt(fd, libc::IPPROTO_IP, libc::IP_MULTICAST_IF, &request as *const _ as *const libc::c_void, std::mem::size_of::<libc::in_addr>() as libc::socklen_t) };
            if r < 0 {
                return Err(map_errno(last_errno()));
            }
            state.multicast_interface = address;
            Ok(())
        }
        IP_RECVDSTADDR | IP_RECVIF | IP_RECVTTL => {
            let number = option_int(ctx, value, len)?;
            let bit = ip_receive_bit(option);
            state.ip_receive_bits = if number != 0 { state.ip_receive_bits | bit } else { state.ip_receive_bits & !bit };
            Ok(())
        }
        IP_TTL | IP_TOS => {
            let number = option_int(ctx, value, len)?;
            let stored = number & 0xFF;
            if option == IP_TOS {
                state.ip_type_of_service = stored;
                return Ok(());
            }
            if stored != 0 {
                set_int_opt(fd, libc::IPPROTO_IP, libc::IP_TTL, stored).map_err(map_errno)?;
            }
            state.ip_time_to_live = stored;
            Ok(())
        }
        IP_MULTICAST_TTL | IP_MULTICAST_LOOP => {
            // A one-byte value is a u_char.
            let number: u32 = if len == 1 {
                if value == 0 {
                    return Err(EFAULT_);
                }
                ctx.read_bytes(value, 1).ok_or(EFAULT_)?[0] as u32
            } else {
                option_int(ctx, value, len)? as u32
            };
            if option == IP_MULTICAST_TTL && number > 255 {
                return Err(EINVAL_);
            }
            let host = if option == IP_MULTICAST_LOOP { ((number & 0xFF) != 0) as i32 } else { number as i32 };
            let native = if option == IP_MULTICAST_LOOP { libc::IP_MULTICAST_LOOP } else { libc::IP_MULTICAST_TTL };
            set_int_opt(fd, libc::IPPROTO_IP, native, host).map_err(map_errno)?;
            if option == IP_MULTICAST_LOOP {
                state.multicast_loop = host != 0;
            } else {
                state.multicast_time_to_live = host;
            }
            Ok(())
        }
        _ => Err(ENOPROTOOPT_),
    }
}

/// The value of an IP-level option from the console's state; a one-byte
/// buffer takes a single byte for the multicast TTL and loop.
fn get_ip_option(state: &SockOptState, ipv4: bool, datagram: bool, option: i32, capacity: u32) -> Result<Vec<u8>, i32> {
    if !ipv4 {
        return Err(EINVAL_);
    }
    if is_multicast_option(option) && !matches!(option, IP_ADD_MEMBERSHIP | IP_DROP_MEMBERSHIP) && !datagram {
        return Err(EOPNOTSUPP_);
    }
    if option == IP_MULTICAST_IF {
        return Ok(state.multicast_interface.to_le_bytes().to_vec());
    }
    let number = if ip_receive_bit(option) != 0 {
        (state.ip_receive_bits & ip_receive_bit(option) != 0) as i32
    } else {
        match option {
            IP_TTL => state.ip_time_to_live,
            IP_TOS => state.ip_type_of_service,
            IP_MULTICAST_TTL => state.multicast_time_to_live,
            IP_MULTICAST_LOOP => state.multicast_loop as i32,
            _ => return Err(ENOPROTOOPT_),
        }
    };
    if matches!(option, IP_MULTICAST_TTL | IP_MULTICAST_LOOP) && capacity == 1 {
        Ok(vec![number as u8])
    } else {
        Ok(number.to_le_bytes().to_vec())
    }
}

/// A FreeBSD socket timeout (sbintime_t: seconds in the high 32 bits, a
/// binary fraction in the low ones) from a timeval, as sbttotv's inverse
/// rounds it; seconds past INT32_MAX saturate.
fn timeout_of(sec: i64, usec: i64) -> i64 {
    if sec > i32::MAX as i64 {
        return i64::MAX;
    }
    (sec << 32) + ((usec as u64).wrapping_mul((1u64 << 63) / 500_000) >> 32) as i64
}

/// The timeval getsockopt reports for a socket timeout.
fn timeval_of(timeout: i64) -> (i64, i64) {
    (timeout >> 32, (1_000_000 * (timeout as u32 as i64)) >> 32)
}

/// The host timeval of a socket timeout: zero waits forever, any other
/// value waits at least a microsecond.
fn host_timeval(timeout: i64) -> libc::timeval {
    if timeout == 0 {
        return libc::timeval { tv_sec: 0, tv_usec: 0 };
    }
    let fraction = timeout as u32 as u64;
    let mut sec = timeout >> 32;
    let mut usec = (fraction * 1_000_000).div_ceil(1u64 << 32) as i64;
    if usec >= 1_000_000 {
        sec += 1;
        usec -= 1_000_000;
    }
    if sec == 0 && usec == 0 {
        usec = 1;
    }
    libc::timeval { tv_sec: sec as libc::time_t, tv_usec: usec as libc::suseconds_t }
}

fn set_host_timeout(fd: i32, option: i32, timeout: i64) -> Result<(), i32> {
    let tv = host_timeval(timeout);
    // SAFETY: one timeval of the given size.
    let r = unsafe {
        libc::setsockopt(fd, libc::SOL_SOCKET, option, &tv as *const _ as *const libc::c_void, std::mem::size_of::<libc::timeval>() as libc::socklen_t)
    };
    if r < 0 { Err(map_errno(last_errno())) } else { Ok(()) }
}

/// O_NONBLOCK of a host socket.
fn set_host_nonblocking(fd: i32, on: bool) -> Result<(), i32> {
    let mut value: libc::c_int = on as libc::c_int;
    // SAFETY: FIONBIO reads one int.
    if unsafe { libc::ioctl(fd, libc::FIONBIO, &mut value) } < 0 { Err(map_errno(last_errno())) } else { Ok(()) }
}

struct KernelSocket {
    /// 1 local (a socketpair end), 2 inet, 28 inet6.
    family: i32,
    /// SOCK_STREAM, SOCK_DGRAM or SOCK_SEQPACKET, without the creation flags.
    ty: i32,
    host: Arc<HostSocket>,
    /// O_NONBLOCK as fcntl reports it (mirrored on the host socket).
    nonblocking: bool,
    /// SO_NOSIGPIPE.
    no_sigpipe: bool,
    /// SO_SNDTIMEO / SO_RCVTIMEO (sbintime_t), also applied on the host.
    send_timeout: i64,
    receive_timeout: i64,
    /// SO_SNDBUF / SO_RCVBUF as the guest set them (0: the host's).
    send_buffer: i32,
    receive_buffer: i32,
    send_low_water: i32,
    receive_low_water: i32,
    /// shutdown() of either direction, for fstat's mode bits.
    receive_shut: bool,
    send_shut: bool,
    /// A connection was started (connect, accept or socketpair): the host
    /// socket's state is the guest's from then on.
    connected: bool,
    listening: bool,
    /// The options the console keeps in its own state.
    opts: SockOptState,
    /// The other end of a socketpair; getpeername reports ENOTCONN once
    /// it is closed.
    peer_host: std::sync::Weak<HostSocket>,
}

impl KernelSocket {
    fn new(family: i32, ty: i32, host: HostSocket) -> Self {
        Self {
            family,
            ty,
            host: Arc::new(host),
            nonblocking: false,
            no_sigpipe: false,
            send_timeout: 0,
            receive_timeout: 0,
            send_buffer: 0,
            receive_buffer: 0,
            // FreeBSD's soreserve defaults (MCLBYTES and 1).
            send_low_water: 2048,
            receive_low_water: 1,
            receive_shut: false,
            send_shut: false,
            connected: false,
            listening: false,
            opts: SockOptState::for_family(family),
            peer_host: Default::default(),
        }
    }

    /// The host descriptor for readiness checks, kept open by the returned
    /// handle. A stream socket that is neither connecting, connected nor
    /// listening has none: FreeBSD reports no event on it (the host would
    /// report a hang-up).
    fn host_handle(&self) -> Option<HostHandle> {
        if self.ty == SOCK_STREAM_K && self.family != AF_UNIX_K && !self.connected && !self.listening {
            return None;
        }
        Some(HostHandle(self.host.clone()))
    }

}

/// A kernel socket's host descriptor, kept open while it is in use outside
/// the socket table lock.
struct HostHandle(Arc<HostSocket>);

impl HostHandle {
    fn fd(&self) -> i32 {
        self.0.fd
    }
}

/// The host descriptor of a kernel socket for poll(), open while held.
pub(crate) struct PollHandle(HostHandle);

impl PollHandle {
    pub(crate) fn fd(&self) -> i32 {
        self.0.fd()
    }
}

/// Host descriptor of a kernel socket for poll(), None if `fd` is not a
/// kernel socket with one.
pub(crate) fn kernel_socket_poll_handle(fd: i32) -> Option<PollHandle> {
    ksockets().get(&fd)?.host_handle().map(PollHandle)
}

/// Host descriptor of a libSceNet socket (`sceNetSocket` id) for poll():
/// None if `id` is no such socket, Err for a P2P socket, whose readiness is
/// per virtual port and not its shared transport's.
pub(crate) fn net_socket_poll_handle(id: i32) -> Option<Result<NetPollHandle, i32>> {
    let st = net();
    let slot = st.sockets.get(&id)?;
    if slot.p2p.is_some() {
        return Some(Err(EOPNOTSUPP_));
    }
    Some(Ok(NetPollHandle(slot.transport.clone())))
}

/// Keeps a libSceNet socket's host descriptor open while it is polled.
pub(crate) struct NetPollHandle(Arc<HostSocket>);

impl NetPollHandle {
    pub(crate) fn fd(&self) -> i32 {
        self.0.fd
    }
}

/// Event-queue read and write events on kernel sockets.
/// `RPS5E_SOCKET_EVENTS=0` turns them off: sockets are then unknown to
/// `sceKernelAddReadEvent` / `sceKernelAddWriteEvent` (ENOENT).
pub(crate) fn socket_events_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_SOCKET_EVENTS").is_ok_and(|v| v.trim() == "0"))
}

/// Readiness of the kernel socket `fd` for an event queue's read (`write`
/// false) or write filter, None when `fd` is not a kernel socket. Read
/// readiness reports the bytes available (pending connections on a
/// listener), write readiness the free send space; a hang-up or socket
/// error adds EV_EOF. A stream socket without a connection is never ready.
pub(crate) fn socket_readiness(fd: i32, write: bool) -> Option<crate::sync::equeue::ReadReadiness> {
    use crate::sync::equeue::{ReadReadiness, EV_EOF};
    if !socket_events_enabled() {
        return None;
    }
    let (handle, listening) = {
        let socks = ksockets();
        let s = socks.get(&fd)?;
        (s.host_handle(), s.listening)
    };
    let Some(handle) = handle else { return Some(ReadReadiness { ready: false, available: 0, flags: 0 }) };
    let raw = handle.fd();
    let events = if write { libc::POLLOUT } else { libc::POLLIN | libc::POLLRDHUP };
    let mut entry = libc::pollfd { fd: raw, events, revents: 0 };
    // SAFETY: one valid pollfd; `handle` keeps the descriptor open.
    if unsafe { libc::poll(&mut entry, 1, 0) } <= 0 {
        return Some(ReadReadiness { ready: false, available: 0, flags: 0 });
    }
    let mut hangup = libc::POLLHUP | libc::POLLERR;
    if !write {
        hangup |= libc::POLLRDHUP;
    }
    let eof = !listening && entry.revents & hangup != 0;
    let available = if write {
        // Free send space: the buffer less what is still queued (SIOCOUTQ).
        let size = get_int_opt(raw, libc::SOL_SOCKET, libc::SO_SNDBUF).unwrap_or(0).max(0) as u64;
        let mut queued: libc::c_int = 0;
        // SAFETY: TIOCOUTQ (SIOCOUTQ) stores one int.
        let queued = if unsafe { libc::ioctl(raw, libc::TIOCOUTQ, &mut queued) } == 0 { queued.max(0) as u64 } else { 0 };
        size.saturating_sub(queued)
    } else if listening {
        listen_queue_length(raw).filter(|&n| n > 0).unwrap_or(1)
    } else {
        let mut n: libc::c_int = 0;
        // SAFETY: FIONREAD stores one int.
        if unsafe { libc::ioctl(raw, libc::FIONREAD, &mut n) } == 0 { n.max(0) as u64 } else { 0 }
    };
    Some(ReadReadiness { ready: true, available, flags: if eof { EV_EOF } else { 0 } })
}

/// Connections waiting in a listening socket's accept queue, which FreeBSD
/// reports as the read event's data (Linux keeps it in tcpi_unacked).
fn listen_queue_length(raw: i32) -> Option<u64> {
    // SAFETY: tcp_info is plain data; getsockopt fills at most `len` bytes.
    unsafe {
        let mut info: libc::tcp_info = std::mem::zeroed();
        let mut len = std::mem::size_of::<libc::tcp_info>() as libc::socklen_t;
        if libc::getsockopt(raw, libc::IPPROTO_TCP, libc::TCP_INFO, (&mut info as *mut libc::tcp_info).cast(), &mut len) != 0 {
            return None;
        }
        Some(info.tcpi_unacked as u64)
    }
}

/// Set O_NONBLOCK of a kernel socket (fcntl F_SETFL, ioctl FIONBIO,
/// SO_NBIO). None if `fd` is not a kernel socket.
pub(crate) fn kernel_socket_set_nonblocking(fd: i32, on: bool) -> Option<Result<(), i32>> {
    let mut socks = ksockets();
    let s = socks.get_mut(&fd)?;
    Some(set_host_nonblocking(s.host.fd, on).map(|()| s.nonblocking = on))
}

/// fcntl() on a kernel socket: F_GETFL reports O_RDWR plus O_NONBLOCK,
/// F_SETFL sets or clears O_NONBLOCK and refuses any other status flag
/// (EOPNOTSUPP); other commands are EINVAL. None if `fd` is not a kernel
/// socket.
pub(crate) fn kernel_socket_fcntl(fd: i32, cmd: u64, arg: u64) -> Option<Result<i64, i32>> {
    const F_GETFL: u64 = 3;
    const F_SETFL: u64 = 4;
    const O_RDWR: i64 = 2;
    const O_NONBLOCK: u64 = 4;
    let nonblocking = ksockets().get(&fd)?.nonblocking;
    Some(match cmd {
        F_GETFL => Ok(O_RDWR | if nonblocking { O_NONBLOCK as i64 } else { 0 }),
        F_SETFL if arg as u32 & !7 != 0 => Err(EOPNOTSUPP_),
        F_SETFL => kernel_socket_set_nonblocking(fd, arg & O_NONBLOCK != 0)?.map(|()| 0),
        _ => Err(EINVAL_),
    })
}

/// Bytes queued for reading on a kernel socket (FIONREAD); None if `fd` is
/// no kernel socket.
pub(crate) fn kernel_socket_queued(fd: i32) -> Option<Result<i64, i32>> {
    let host = ksockets().get(&fd)?.host.clone();
    let mut n: libc::c_int = 0;
    // SAFETY: FIONREAD writes one int to the valid pointer.
    let r = unsafe { libc::ioctl(host.fd, libc::FIONREAD, &mut n) };
    Some(if r < 0 { Err(map_errno(last_errno())) } else { Ok(n as i64) })
}

/// What fstat reports for a socket.
pub(crate) struct SocketStat {
    /// S_IFSOCK with read bits while data can still arrive (or is queued)
    /// and write bits while sending is open.
    pub(crate) mode: u16,
    /// Bytes queued for reading.
    pub(crate) size: i64,
    /// The send buffer size.
    pub(crate) block_size: u32,
}

/// fstat() of a kernel socket, as FreeBSD's soo_stat fills it; None if
/// `fd` is no kernel socket.
pub(crate) fn kernel_socket_stat(fd: i32) -> Option<SocketStat> {
    let (host, receive_shut, send_shut, send_buffer) = {
        let socks = ksockets();
        let s = socks.get(&fd)?;
        (s.host.clone(), s.receive_shut, s.send_shut, s.send_buffer)
    };
    let mut available: libc::c_int = 0;
    // SAFETY: FIONREAD writes one int.
    if unsafe { libc::ioctl(host.fd, libc::FIONREAD, &mut available) } != 0 {
        available = 0;
    }
    let block = if send_buffer > 0 { send_buffer } else { get_int_opt(host.fd, libc::SOL_SOCKET, libc::SO_SNDBUF).unwrap_or(0) };
    let mut mode = 0o140000u16;
    if !receive_shut || available > 0 {
        mode |= 0o444;
    }
    if !send_shut {
        mode |= 0o222;
    }
    Some(SocketStat { mode, size: available.max(0) as i64, block_size: block.max(0) as u32 })
}

static KERNEL_SOCKETS: LazyLock<Mutex<HashMap<i32, KernelSocket>>> = LazyLock::new(|| Mutex::new(HashMap::new()));

fn ksockets() -> MutexGuard<'static, HashMap<i32, KernelSocket>> {
    KERNEL_SOCKETS.lock().unwrap_or_else(|p| p.into_inner())
}

pub(crate) fn is_kernel_socket(fd: i32) -> bool {
    ksockets().contains_key(&fd)
}

/// Type of the kernel socket `fd` (1 stream, 2 datagram, ...).
pub(crate) fn kernel_socket_type(fd: i32) -> Option<i32> {
    ksockets().get(&fd).map(|s| s.ty)
}

/// Address family of the kernel socket `fd` (1 local, 2 inet, 28 inet6).
pub(crate) fn kernel_socket_family(fd: i32) -> Option<i32> {
    ksockets().get(&fd).map(|s| s.family)
}

/// True when `fd` is a listening kernel socket.
pub(crate) fn is_listening_socket(fd: i32) -> bool {
    ksockets().get(&fd).is_some_and(|s| s.listening)
}

/// Put a new kernel socket on the lowest free descriptor number.
fn install_socket(sock: KernelSocket) -> i32 {
    let fd = crate::filesystem::reserve_fd();
    ksockets().insert(fd, sock);
    fd
}

/// close() hook: true when `fd` was a kernel socket (now closed and its
/// number free).
pub(crate) fn kernel_socket_close(fd: i32) -> bool {
    let removed = ksockets().remove(&fd);
    match removed {
        Some(_) => {
            crate::filesystem::release_fd(fd);
            if socket_events_enabled() {
                crate::sync::equeue::descriptor_closed(fd);
            }
            true
        }
        None => false,
    }
}

/// The errno for a descriptor that is not a kernel socket: ENOTSOCK when
/// it names something else, EBADF when nothing.
fn not_a_socket(fd: i32) -> i32 {
    if crate::filesystem::descriptor_open(fd) { ENOTSOCK_ } else { EBADF_ }
}

/// What a socket call needs of a kernel socket, taken under the table lock.
struct SocketRef {
    host: Arc<HostSocket>,
    family: i32,
    ty: i32,
    connected: bool,
    listening: bool,
    /// The other end of a socketpair, for getpeername's ENOTCONN.
    peer_host: std::sync::Weak<HostSocket>,
}

impl SocketRef {
    /// TCP options apply to inet stream sockets only.
    fn is_tcp(&self) -> bool {
        self.family != AF_UNIX_K && self.ty == SOCK_STREAM_K
    }
}

fn socket_ref(fd: i32) -> Result<SocketRef, i32> {
    let found = ksockets()
        .get(&fd)
        .map(|s| SocketRef { host: s.host.clone(), family: s.family, ty: s.ty, connected: s.connected, listening: s.listening, peer_host: s.peer_host.clone() });
    found.ok_or_else(|| not_a_socket(fd))
}

/// read() on a kernel socket: recv() without flags. None = `fd` is not a
/// kernel socket.
pub(crate) fn kernel_socket_read(ctx: &CpuContext, fd: i32, buf: u64, requested: usize) -> Option<Result<i64, i32>> {
    Some(match kernel_socket_recv_bytes(fd, requested)? {
        Ok(data) if !data.is_empty() && !ctx.write_bytes(buf, &data) => Err(EFAULT_),
        Ok(data) => Ok(data.len() as i64),
        Err(e) => Err(e),
    })
}

/// One recv() without flags of up to `requested` bytes on a kernel socket,
/// into host memory (a scatter read keeps the datagram or stream unit of a
/// single read). None = `fd` is not a kernel socket.
pub(crate) fn kernel_socket_recv_bytes(fd: i32, requested: usize) -> Option<Result<Vec<u8>, i32>> {
    let s = socket_ref(fd).ok()?;
    let mut data = vec![0u8; requested];
    Some(host_receive(&s, &mut data, 0).map(|(n, _)| {
        data.truncate(n);
        data
    }))
}

/// A connected AF_UNIX kernel socket pair of type `ty` (1 stream, 2
/// datagram), for tests of other modules; `sv` is guest scratch for the
/// two descriptors.
#[cfg(test)]
pub(crate) fn test_socketpair(ctx: &CpuContext, ty: i32, sv: u64) -> (i32, i32) {
    posix_ext::test_socketpair(ctx, ty, sv)
}

/// One send() without flags of host bytes on a kernel socket (a gather
/// write keeps one datagram). None = `fd` is not a kernel socket.
pub(crate) fn kernel_socket_send_bytes(fd: i32, data: &[u8]) -> Option<Result<i64, i32>> {
    let s = socket_ref(fd).ok()?;
    Some(host_send(&s, data, 0, None))
}

/// write() on a kernel socket: send() without flags (EPIPE once the peer
/// is gone). None = `fd` is not a kernel socket.
pub(crate) fn kernel_socket_write(ctx: &CpuContext, fd: i32, buf: u64, requested: usize) -> Option<Result<i64, i32>> {
    if !is_kernel_socket(fd) {
        return None;
    }
    Some(send_core(ctx, fd, buf, requested as u64, 0, 0, 0))
}

fn posix_socket_failure(ctx: &CpuContext, errno: i32) -> HleResult {
    crate::kernel::set_errno(ctx, errno);
    -1
}

fn posix_socket_result(ctx: &CpuContext, r: Result<i64, i32>) -> HleResult {
    match r {
        Ok(v) => v,
        Err(e) => posix_socket_failure(ctx, e),
    }
}

/// Destination rewrite configured by RPS5E_NET_REDIRECT.
fn net_redirect() -> Option<IpAddr> {
    let v = std::env::var("RPS5E_NET_REDIRECT").ok()?;
    let v = v.trim();
    if v.is_empty() {
        return None;
    }
    parse_ip_literal(v)
}

fn net_redirect_configured() -> bool {
    std::env::var("RPS5E_NET_REDIRECT").is_ok_and(|v| !v.trim().is_empty())
}

/// Outbound traffic is allowed to loopback, or anywhere once a redirect is
/// configured.
fn outbound_allowed(ip: IpAddr, redirected: bool) -> bool {
    redirected || net_redirect_configured() || ip.is_loopback()
}

/// The host destination of guest traffic to `dest` on `fd`: rewritten by
/// the redirect, and refused (ENETUNREACH) while offline unless it is
/// loopback or the redirect's address does not fit the socket's family.
fn outbound_destination_for(fd: i32, dest: SocketAddr) -> Result<SocketAddr, i32> {
    let redirect = net_redirect();
    let ip = redirect.unwrap_or(dest.ip());
    if !outbound_allowed(ip, redirect.is_some()) {
        log::info!("[net] fd={fd}: traffic to {dest} denied (offline)");
        return Err(ENETUNREACH_);
    }
    if ip.is_ipv4() != dest.is_ipv4() {
        log::error!("[net] fd={fd}: redirect {ip} does not fit a {} destination", if dest.is_ipv4() { "IPv4" } else { "IPv6" });
        return Err(ENETUNREACH_);
    }
    Ok(match dest {
        SocketAddr::V6(v6) => match ip {
            IpAddr::V6(a) => SocketAddr::V6(std::net::SocketAddrV6::new(a, v6.port(), v6.flowinfo(), v6.scope_id())),
            IpAddr::V4(_) => dest,
        },
        SocketAddr::V4(_) => SocketAddr::new(ip, dest.port()),
    })
}

/// Guest sockaddr at `addr` with length `len` -> host address, checked as
/// FreeBSD 11's getsockaddr and the protocols do: a length above 255 is
/// ENAMETOOLONG, below 2 EINVAL, a null address EFAULT; an AF_INET address
/// shorter than 16 bytes or an AF_INET6 one shorter than 28 is EINVAL, any
/// other family EAFNOSUPPORT.
fn read_socket_address(ctx: &CpuContext, addr: u64, len: u32) -> Result<SocketAddr, i32> {
    const SOCK_MAXADDRLEN: u32 = 255;
    const ENAMETOOLONG_: i32 = 63;
    if len > SOCK_MAXADDRLEN {
        return Err(ENAMETOOLONG_);
    }
    if len < 2 {
        return Err(EINVAL_);
    }
    if addr == 0 {
        return Err(EFAULT_);
    }
    let head = ctx.read_bytes(addr, 2).ok_or(EFAULT_)?;
    match head[1] as i32 {
        ORBIS_AF_INET if len < 16 => Err(EINVAL_),
        ORBIS_AF_INET6 if len < 28 => Err(EINVAL_),
        ORBIS_AF_INET if len >= 16 => {
            let b = ctx.read_bytes(addr, 8).ok_or(EFAULT_)?;
            Ok(SocketAddr::new(IpAddr::V4(Ipv4Addr::new(b[4], b[5], b[6], b[7])), u16::from_be_bytes([b[2], b[3]])))
        }
        ORBIS_AF_INET6 if len >= 28 => {
            let b = ctx.read_bytes(addr, 28).ok_or(EFAULT_)?;
            let ip: [u8; 16] = b[8..24].try_into().unwrap();
            Ok(SocketAddr::V6(std::net::SocketAddrV6::new(
                Ipv6Addr::from(ip),
                u16::from_be_bytes([b[2], b[3]]),
                u32::from_ne_bytes(b[4..8].try_into().unwrap()),
                u32::from_ne_bytes(b[24..28].try_into().unwrap()),
            )))
        }
        _ => Err(EAFNOSUPPORT_),
    }
}

/// The address of a bind, connect or sendto on a socket of `family`, as a
/// PS5 (firmware 13.60) checks it on datagram sockets: past the generic
/// length rules an AF_INET socket takes exactly 16 bytes and an AF_INET6
/// socket exactly 28 (else EINVAL, checked before the family), another
/// family is EAFNOSUPPORT. Stream sockets follow the same order.
fn read_socket_address_for(ctx: &CpuContext, family: i32, addr: u64, len: u32) -> Result<SocketAddr, i32> {
    if family != ORBIS_AF_INET && family != ORBIS_AF_INET6 {
        return read_socket_address(ctx, addr, len);
    }
    check_socket_address(addr, len)?;
    let head = ctx.read_bytes(addr, 2).ok_or(EFAULT_)?;
    if len != if family == ORBIS_AF_INET { 16 } else { 28 } {
        return Err(EINVAL_);
    }
    if head[1] as i32 != family {
        return Err(EAFNOSUPPORT_);
    }
    read_socket_address(ctx, addr, len)
}

/// The generic socket address rules (FreeBSD's getsockaddr): longer than
/// SOCK_MAXADDRLEN is ENAMETOOLONG, shorter than 2 bytes EINVAL, NULL
/// EFAULT.
fn check_socket_address(addr: u64, len: u32) -> Result<(), i32> {
    const SOCK_MAXADDRLEN: u32 = 255;
    const ENAMETOOLONG_: i32 = 63;
    if len > SOCK_MAXADDRLEN {
        return Err(ENAMETOOLONG_);
    }
    if len < 2 {
        return Err(EINVAL_);
    }
    if addr == 0 {
        return Err(EFAULT_);
    }
    Ok(())
}

/// Whether `addr` is of the socket's family (inet sockets take AF_INET,
/// inet6 sockets AF_INET6).
fn family_fits(family: i32, addr: &SocketAddr) -> bool {
    (family == ORBIS_AF_INET) == addr.is_ipv4()
}

/// The host address of a socket (getsockname or getpeername).
fn host_name(fd: i32, peer: bool) -> Result<SocketAddr, i32> {
    let mut ss: libc::sockaddr_storage = unsafe { std::mem::zeroed() };
    let mut len = std::mem::size_of::<libc::sockaddr_storage>() as libc::socklen_t;
    let p = &mut ss as *mut _ as *mut libc::sockaddr;
    // SAFETY: the storage holds any address.
    let r = unsafe { if peer { libc::getpeername(fd, p, &mut len) } else { libc::getsockname(fd, p, &mut len) } };
    if r < 0 {
        return Err(map_errno(last_errno()));
    }
    sockaddr_to(&ss, len).ok_or(EAFNOSUPPORT_)
}

fn host_connect(fd: i32, addr: &SocketAddr) -> Result<(), i32> {
    let (ss, len) = sockaddr_from(addr);
    // SAFETY: a valid address of `len` bytes.
    if unsafe { libc::connect(fd, &ss as *const _ as *const libc::sockaddr, len) } < 0 {
        return Err(map_errno(last_errno()));
    }
    Ok(())
}

fn host_bind(fd: i32, addr: &SocketAddr) -> Result<(), i32> {
    let (ss, len) = sockaddr_from(addr);
    // SAFETY: a valid address of `len` bytes.
    if unsafe { libc::bind(fd, &ss as *const _ as *const libc::sockaddr, len) } < 0 {
        return Err(map_errno(last_errno()));
    }
    Ok(())
}

/// Host flags of a send: MSG_DONTROUTE, MSG_DONTWAIT and MSG_NOSIGNAL are
/// known (EOPNOTSUPP otherwise); the host never raises SIGPIPE.
fn send_host_flags(flags: i32) -> Result<i32, i32> {
    if flags & !(MSG_DONTROUTE | MSG_DONTWAIT | MSG_NOSIGNAL) != 0 {
        return Err(EOPNOTSUPP_);
    }
    let mut host = libc::MSG_NOSIGNAL;
    for (guest, h) in [(MSG_DONTROUTE, libc::MSG_DONTROUTE), (MSG_DONTWAIT, libc::MSG_DONTWAIT)] {
        if flags & guest != 0 {
            host |= h;
        }
    }
    Ok(host)
}

/// Host flags of a receive: MSG_PEEK, MSG_DONTROUTE, MSG_WAITALL,
/// MSG_DONTWAIT and MSG_NOSIGNAL are known (EOPNOTSUPP otherwise).
fn receive_host_flags(flags: i32) -> Result<i32, i32> {
    if flags & !(MSG_PEEK | MSG_DONTROUTE | MSG_WAITALL | MSG_DONTWAIT | MSG_NOSIGNAL) != 0 {
        return Err(EOPNOTSUPP_);
    }
    let mut host = 0;
    for (guest, h) in [(MSG_PEEK, libc::MSG_PEEK), (MSG_WAITALL, libc::MSG_WAITALL), (MSG_DONTWAIT, libc::MSG_DONTWAIT)] {
        if flags & guest != 0 {
            host |= h;
        }
    }
    Ok(host)
}

/// One host send (or sendto `dest`) of `data` with checked host flags.
/// A stream socket that never connected reports ENOTCONN as FreeBSD does
/// (the host says EPIPE).
fn host_send(s: &SocketRef, data: &[u8], host_flags: i32, dest: Option<SocketAddr>) -> Result<i64, i32> {
    let host_flags = host_flags | libc::MSG_NOSIGNAL;
    let r = match dest {
        Some(d) => {
            let (ss, len) = sockaddr_from(&d);
            retry(|| unsafe { libc::sendto(s.host.fd, data.as_ptr() as *const libc::c_void, data.len(), host_flags, &ss as *const _ as *const libc::sockaddr, len) })
        }
        None => retry(|| unsafe { libc::send(s.host.fd, data.as_ptr() as *const libc::c_void, data.len(), host_flags) }),
    };
    match r {
        Ok(n) => Ok(n as i64),
        Err(libc::EPIPE) if s.ty == SOCK_STREAM_K && !s.connected => Err(ENOTCONN_),
        Err(e) => Err(map_errno(e)),
    }
}

/// One host receive into `data`; also returns the sender, if the host
/// reported one.
fn host_receive(s: &SocketRef, data: &mut [u8], host_flags: i32) -> Result<(usize, Option<SocketAddr>), i32> {
    let mut ss: libc::sockaddr_storage = unsafe { std::mem::zeroed() };
    let mut len: libc::socklen_t = 0;
    let n = retry(|| unsafe {
        len = std::mem::size_of::<libc::sockaddr_storage>() as libc::socklen_t;
        libc::recvfrom(s.host.fd, data.as_mut_ptr() as *mut libc::c_void, data.len(), host_flags, &mut ss as *mut _ as *mut libc::sockaddr, &mut len)
    })
    .map_err(map_errno)?;
    Ok((n, sockaddr_to(&ss, len)))
}

/// Where a send goes: `addr` 0 sends on the connection (a non-zero
/// `addrlen` with it is EINVAL); otherwise the datagram goes to that
/// address, which must fit the socket's family and the outbound policy. A
/// connected datagram socket refuses an address (EISCONN); a stream socket
/// ignores it.
fn send_destination(ctx: &CpuContext, fd: i32, s: &SocketRef, addr: u64, addrlen: u32) -> Result<Option<SocketAddr>, i32> {
    if addr == 0 {
        return if addrlen != 0 { Err(EINVAL_) } else { Ok(None) };
    }
    // A connected stream socket ignores the address once its generic
    // length rules pass (FreeBSD's tcp_usr_send).
    if s.ty == SOCK_STREAM_K && s.family != AF_UNIX_K && host_name(s.host.fd, true).is_ok() {
        check_socket_address(addr, addrlen)?;
        return Ok(None);
    }
    let dest = read_socket_address_for(ctx, s.family, addr, addrlen)?;
    if !family_fits(s.family, &dest) {
        return Err(EAFNOSUPPORT_);
    }
    match s.ty {
        SOCK_STREAM_K => Ok(None),
        _ if host_name(s.host.fd, true).is_ok() => Err(EISCONN_),
        _ => outbound_destination_for(fd, dest).map(Some),
    }
}

/// send / sendto core (see send_destination). Longer than INT_MAX is
/// EMSGSIZE.
fn send_core(ctx: &CpuContext, fd: i32, buf: u64, length: u64, flags: i32, addr: u64, addrlen: u32) -> Result<i64, i32> {
    let s = socket_ref(fd)?;
    if length > i32::MAX as u64 {
        return Err(EMSGSIZE_);
    }
    if length != 0 && buf == 0 {
        return Err(EFAULT_);
    }
    let host_flags = send_host_flags(flags)?;
    let dest = send_destination(ctx, fd, &s, addr, addrlen)?;
    let data = if length == 0 { Vec::new() } else { ctx.read_bytes(buf, length as usize).ok_or(EFAULT_)? };
    host_send(&s, &data, host_flags, dest)
}

/// recv / recvfrom core. With `addr`, the sender is written as much as
/// `*addrlen` holds: none on a stream socket (`*addrlen` 0), the unnamed
/// address on a local datagram socket.
fn receive_core(ctx: &CpuContext, fd: i32, buf: u64, length: u64, flags: i32, addr: u64, addrlen: u64) -> Result<i64, i32> {
    let s = socket_ref(fd)?;
    if length > i32::MAX as u64 {
        return Err(EMSGSIZE_);
    }
    if (length != 0 && buf == 0) || (addr != 0 && addrlen == 0) {
        return Err(EFAULT_);
    }
    let host_flags = receive_host_flags(flags)?;
    let mut data = vec![0u8; length as usize];
    let (n, sender) = host_receive(&s, &mut data, host_flags)?;
    if n != 0 && !ctx.write_bytes(buf, &data[..n]) {
        return Err(EFAULT_);
    }
    if addr != 0 {
        if s.family == AF_UNIX_K && s.ty != SOCK_STREAM_K {
            posix_ext::write_unnamed_address(ctx, addr, addrlen)?;
        } else {
            let sender = if s.family == AF_UNIX_K { None } else { sender };
            convert_host_sockaddr(ctx, sender, addr, addrlen)?;
        }
    }
    Ok(n as i64)
}

/// socket(domain, type, protocol) -> fd. AF_INET and AF_INET6 stream (TCP)
/// and datagram (UDP) sockets; SOCK_NONBLOCK and SOCK_CLOEXEC may be or-ed
/// into the type. A type inet does not have is EPROTOTYPE (raw and
/// sequenced-packet ones, which need a protocol, EPROTONOSUPPORT).
fn kernel_socket_core(family: i32, ty: i32, protocol: i32) -> Result<i32, i32> {
    let nonblocking = ty & SOCK_NONBLOCK_K != 0;
    let cloexec = ty & SOCK_CLOEXEC_K != 0;
    let plain = ty & !(SOCK_CLOEXEC_K | SOCK_NONBLOCK_K);
    if family != ORBIS_AF_INET && family != ORBIS_AF_INET6 {
        return Err(EAFNOSUPPORT_);
    }
    let native = match plain {
        SOCK_STREAM_K => libc::SOCK_STREAM,
        SOCK_DGRAM_K => libc::SOCK_DGRAM,
        _ => {
            let inet_knows = plain == 3 || plain == 5;
            return Err(if protocol == 0 && plain != 0 && !inet_knows { EPROTOTYPE_ } else { EPROTONOSUPPORT_ });
        }
    };
    let expected = if plain == SOCK_STREAM_K { 6 } else { 17 };
    if protocol != 0 && protocol != expected {
        return Err(EPROTONOSUPPORT_);
    }
    let domain = if family == ORBIS_AF_INET { libc::AF_INET } else { libc::AF_INET6 };
    let host = HostSocket::new(domain, native | if nonblocking { libc::SOCK_NONBLOCK } else { 0 }, protocol).map_err(map_errno)?;
    if family == ORBIS_AF_INET6 {
        set_int_opt(host.fd, libc::IPPROTO_IPV6, libc::IPV6_V6ONLY, 1).map_err(map_errno)?;
    }
    let mut sock = KernelSocket::new(family, plain, host);
    sock.nonblocking = nonblocking;
    let fd = install_socket(sock);
    if cloexec {
        crate::filesystem::set_cloexec(fd, true);
    }
    net_log(format_args!("kernel socket fd={fd} family={family} type={ty:#x} protocol={protocol}"));
    Ok(fd)
}

fn cb_kernel_socket(ctx: &mut CpuContext) -> HleResult {
    let r = kernel_socket_core(ctx.rdi() as i32, ctx.rsi() as i32, ctx.rdx() as i32);
    posix_socket_result(ctx, r.map(|fd| fd as i64))
}

/// __sys_socketex(const char* name, family, type, protocol): socket() with
/// a debug name, which is ignored.
fn cb_kernel_socketex(ctx: &mut CpuContext) -> HleResult {
    let r = kernel_socket_core(ctx.rsi() as i32, ctx.rdx() as i32, ctx.rcx() as i32);
    posix_socket_result(ctx, r.map(|fd| fd as i64))
}

/// Dissolve the association of a connected datagram socket (connect to
/// AF_UNSPEC); the local port stays bound, as on FreeBSD.
fn disconnect_datagram(s: &SocketRef) -> Result<(), i32> {
    if host_name(s.host.fd, true).is_err() {
        return Ok(());
    }
    let local = host_name(s.host.fd, false)?;
    let mut unspec: libc::sockaddr = unsafe { std::mem::zeroed() };
    unspec.sa_family = libc::AF_UNSPEC as libc::sa_family_t;
    // SAFETY: a plain sockaddr of its own size.
    if unsafe { libc::connect(s.host.fd, &unspec, std::mem::size_of::<libc::sockaddr>() as libc::socklen_t) } < 0 {
        return Err(map_errno(last_errno()));
    }
    if host_name(s.host.fd, false)?.port() != 0 {
        return Ok(());
    }
    let wildcard = match local {
        SocketAddr::V4(_) => SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), local.port()),
        SocketAddr::V6(_) => SocketAddr::new(IpAddr::V6(Ipv6Addr::UNSPECIFIED), local.port()),
    };
    host_bind(s.host.fd, &wildcard)
}

/// connect(fd, addr, addrlen). On a datagram socket an AF_UNSPEC address
/// first dissolves the association (then fails with EAFNOSUPPORT, as the
/// family is not inet). The address is checked before the socket state: a
/// listening socket refuses a valid one with EOPNOTSUPP. Destinations
/// follow the outbound policy; a non-blocking stream connect reports
/// EINPROGRESS.
fn kernel_connect(ctx: &CpuContext, fd: i32, addr: u64, addrlen: u32) -> Result<(), i32> {
    let s = socket_ref(fd)?;
    if s.family != AF_UNIX_K && s.ty == SOCK_DGRAM_K && addr != 0 && addrlen >= 2 && ctx.read_bytes(addr + 1, 1).is_some_and(|b| b[0] == 0) {
        disconnect_datagram(&s)?;
    }
    // A listening stream socket refuses any address that passes the
    // generic length rules (soconnect checks it before the protocol).
    if s.ty == SOCK_STREAM_K && s.listening {
        check_socket_address(addr, addrlen)?;
        return Err(EOPNOTSUPP_);
    }
    let dest = read_socket_address_for(ctx, s.family, addr, addrlen)?;
    // Port 0 is no destination: FreeBSD's in_pcbconnect_setup fails it with
    // EADDRNOTAVAIL for a stream socket and an unconnected datagram one
    // (Linux would connect a datagram socket, or refuse a stream one).
    if (s.ty == SOCK_STREAM_K || s.ty == SOCK_DGRAM_K)
        && s.family != AF_UNIX_K
        && family_fits(s.family, &dest)
        && dest.port() == 0
        && host_name(s.host.fd, true).is_err()
    {
        return Err(EADDRNOTAVAIL_);
    }
    if s.family == AF_UNIX_K || !family_fits(s.family, &dest) {
        return Err(EAFNOSUPPORT_);
    }
    let target = outbound_destination_for(fd, dest)?;
    if s.ty == SOCK_STREAM_K {
        if let Some(k) = ksockets().get_mut(&fd) {
            k.connected = true;
        }
    }
    let r = host_connect(s.host.fd, &target);
    net_log(format_args!("connect: fd={fd} {dest} -> {r:?}"));
    r
}

fn cb_kernel_connect(ctx: &mut CpuContext) -> HleResult {
    let r = kernel_connect(ctx, ctx.rdi() as i32, ctx.rsi(), ctx.rdx() as u32);
    posix_socket_result(ctx, r.map(|()| 0))
}

/// The loopback address of a socket family.
fn loopback_of(family: i32) -> IpAddr {
    if family == ORBIS_AF_INET6 { IpAddr::V6(Ipv6Addr::LOCALHOST) } else { IpAddr::V4(Ipv4Addr::LOCALHOST) }
}

/// bind(fd, addr, addrlen). Unless an outbound redirect is configured, a
/// stream socket bound to the wildcard address is confined to loopback, so
/// the guest never exposes host ports to the network.
fn kernel_bind(ctx: &CpuContext, fd: i32, addr: u64, addrlen: u32) -> Result<(), i32> {
    let s = socket_ref(fd)?;
    let mut local = read_socket_address_for(ctx, s.family, addr, addrlen)?;
    // The address family must match the socket's own: the console refuses
    // a mismatch with EAFNOSUPPORT (measured on hardware).
    if s.family != AF_UNIX_K && !family_fits(s.family, &local) {
        return Err(EAFNOSUPPORT_);
    }
    if s.ty == SOCK_STREAM_K && local.ip().is_unspecified() && family_fits(s.family, &local) && !net_redirect_configured() {
        local.set_ip(loopback_of(s.family));
    }
    host_bind(s.host.fd, &local)?;
    net_log(format_args!("bind: fd={fd} {local}"));
    Ok(())
}

fn cb_kernel_bind(ctx: &mut CpuContext) -> HleResult {
    let r = kernel_bind(ctx, ctx.rdi() as i32, ctx.rsi(), ctx.rdx() as u32);
    posix_socket_result(ctx, r.map(|()| 0))
}

/// getsockname / getpeername: the unnamed address for a socketpair end,
/// otherwise the host's, written as far as `*len_addr` allows.
fn socket_name(ctx: &CpuContext, fd: i32, addr: u64, len_addr: u64, peer: bool) -> Result<(), i32> {
    let s = socket_ref(fd)?;
    if addr == 0 || len_addr == 0 {
        return Err(EFAULT_);
    }
    if s.family == AF_UNIX_K {
        // A socketpair end whose peer is gone is no longer connected
        // (measured on hardware).
        if peer && s.peer_host.upgrade().is_none() {
            return Err(ENOTCONN_);
        }
        return posix_ext::write_unnamed_address(ctx, addr, len_addr);
    }
    let name = host_name(s.host.fd, peer)?;
    convert_host_sockaddr(ctx, Some(name), addr, len_addr).map(|_| ())
}

fn cb_kernel_getsockname(ctx: &mut CpuContext) -> HleResult {
    let r = socket_name(ctx, ctx.rdi() as i32, ctx.rsi(), ctx.rdx(), false);
    posix_socket_result(ctx, r.map(|()| 0))
}

/// Read the int value of a set option: shorter than an int is EINVAL, a
/// longer value is read from its first int.
fn option_int(ctx: &CpuContext, value: u64, len: u32) -> Result<i32, i32> {
    if len < 4 || value == 0 {
        return Err(EINVAL_);
    }
    ctx.read_u32(value).map(|v| v as i32).ok_or(EFAULT_)
}

/// setsockopt on a kernel socket: SOL_SOCKET (flags, buffers, low-water
/// marks, linger, timeouts, SO_NOSIGPIPE, SO_NBIO), IPPROTO_TCP
/// (TCP_NODELAY on inet stream sockets) and IPPROTO_IPV6 (IPV6_V6ONLY on
/// inet6 sockets); anything else is ENOPROTOOPT.
fn kernel_setsockopt(ctx: &CpuContext, fd: i32, level: i32, option: i32, value: u64, len: u32) -> Result<(), i32> {
    let s = socket_ref(fd)?;
    let host = s.host.fd;
    let host_err = |e: i32| map_errno(e);
    match level {
        SOL_SOCKET => {}
        IPPROTO_TCP => {
            if !s.is_tcp() {
                return Err(EINVAL_);
            }
            if !tcp_option_known(option) {
                return Err(ENOPROTOOPT_);
            }
            let number = option_int(ctx, value, len)?;
            if option == TCP_NODELAY {
                return set_int_opt(host, libc::IPPROTO_TCP, libc::TCP_NODELAY, (number != 0) as i32).map_err(host_err);
            }
            let mut opts = ksockets().get(&fd).map(|k| k.opts.clone()).ok_or(EBADF_)?;
            set_tcp_state_option(host, &mut opts, option, number)?;
            if let Some(k) = ksockets().get_mut(&fd) {
                k.opts = opts;
            }
            return Ok(());
        }
        IPPROTO_IP => {
            let ipv4 = s.family == ORBIS_AF_INET;
            let datagram = s.ty == SOCK_DGRAM_K;
            let mut opts = ksockets().get(&fd).map(|k| k.opts.clone()).ok_or(EBADF_)?;
            set_ip_option(ctx, host, &mut opts, ipv4, datagram, option, value, len)?;
            if let Some(k) = ksockets().get_mut(&fd) {
                k.opts = opts;
            }
            return Ok(());
        }
        IPPROTO_IPV6 if option == IPV6_V6ONLY && s.family == ORBIS_AF_INET6 => {
            let v = option_int(ctx, value, len)?;
            return set_int_opt(host, libc::IPPROTO_IPV6, libc::IPV6_V6ONLY, (v != 0) as i32).map_err(host_err);
        }
        _ => return Err(ENOPROTOOPT_),
    }
    if kept_or_write_only(option) {
        let v = option_int(ctx, value, len)?;
        let mut opts = ksockets().get(&fd).map(|k| k.opts.clone()).ok_or(EBADF_)?;
        set_kept_flag_option(host, &mut opts, option, v)?;
        if let Some(k) = ksockets().get_mut(&fd) {
            k.opts = opts;
        }
        return Ok(());
    }
    if let Some(&(_, native)) = FLAG_OPTIONS.iter().find(|(o, _)| *o == option) {
        let v = option_int(ctx, value, len)?;
        return set_int_opt(host, libc::SOL_SOCKET, native, (v != 0) as i32).map_err(host_err);
    }
    match option {
        SO_LINGER => {
            if len < 8 || value == 0 {
                return Err(EINVAL_);
            }
            let b = ctx.read_bytes(value, 8).ok_or(EFAULT_)?;
            let on = i32::from_le_bytes(b[0..4].try_into().unwrap());
            let seconds = i32::from_le_bytes(b[4..8].try_into().unwrap());
            if !(0..=MAX_LINGER).contains(&seconds) {
                return Err(EDOM_);
            }
            let l = libc::linger { l_onoff: (on != 0) as i32, l_linger: seconds };
            // SAFETY: one linger of its own size.
            let r = unsafe { libc::setsockopt(host, libc::SOL_SOCKET, libc::SO_LINGER, &l as *const _ as *const libc::c_void, std::mem::size_of::<libc::linger>() as libc::socklen_t) };
            if r < 0 { Err(map_errno(last_errno())) } else { Ok(()) }
        }
        SO_SNDTIMEO | SO_RCVTIMEO => {
            if len < GUEST_TIMEVAL_SIZE || value == 0 {
                return Err(EINVAL_);
            }
            let b = ctx.read_bytes(value, 16).ok_or(EFAULT_)?;
            let sec = i64::from_le_bytes(b[0..8].try_into().unwrap());
            let usec = i64::from_le_bytes(b[8..16].try_into().unwrap());
            if sec < 0 || !(0..1_000_000).contains(&usec) {
                return Err(EDOM_);
            }
            let timeout = timeout_of(sec, usec);
            let receive = option == SO_RCVTIMEO;
            set_host_timeout(host, if receive { libc::SO_RCVTIMEO } else { libc::SO_SNDTIMEO }, timeout)?;
            if let Some(k) = ksockets().get_mut(&fd) {
                if receive {
                    k.receive_timeout = timeout;
                } else {
                    k.send_timeout = timeout;
                }
            }
            Ok(())
        }
        SO_NOSIGPIPE => {
            let v = option_int(ctx, value, len)?;
            if let Some(k) = ksockets().get_mut(&fd) {
                k.no_sigpipe = v != 0;
            }
            Ok(())
        }
        SO_NBIO => {
            let v = option_int(ctx, value, len)?;
            kernel_socket_set_nonblocking(fd, v != 0).unwrap_or(Err(EBADF_))
        }
        SO_SNDBUF | SO_RCVBUF => {
            let v = option_int(ctx, value, len)?;
            if v < 1 {
                return Err(EINVAL_);
            }
            let send = option == SO_SNDBUF;
            set_int_opt(host, libc::SOL_SOCKET, if send { libc::SO_SNDBUF } else { libc::SO_RCVBUF }, v).map_err(host_err)?;
            if let Some(k) = ksockets().get_mut(&fd) {
                if send {
                    k.send_buffer = v;
                } else {
                    k.receive_buffer = v;
                }
            }
            Ok(())
        }
        SO_SNDLOWAT | SO_RCVLOWAT => {
            let v = option_int(ctx, value, len)?;
            if v < 1 {
                return Err(EINVAL_);
            }
            if option == SO_RCVLOWAT {
                if v > 1 && s.ty != SOCK_STREAM_K {
                    log::error!("[net] setsockopt SO_RCVLOWAT {v} on a socket that is not a stream: not supported");
                    return Err(EINVAL_);
                }
                set_int_opt(host, libc::SOL_SOCKET, libc::SO_RCVLOWAT, v).map_err(host_err)?;
            }
            if let Some(k) = ksockets().get_mut(&fd) {
                if option == SO_RCVLOWAT {
                    k.receive_low_water = v;
                } else {
                    k.send_low_water = v;
                }
            }
            Ok(())
        }
        _ => {
            net_log(format_args!("setsockopt unknown option: fd={fd} level={level:#x} option={option:#x}"));
            Err(ENOPROTOOPT_)
        }
    }
}

/// getsockopt on a kernel socket (see kernel_setsockopt); also SO_TYPE,
/// SO_ACCEPTCONN and SO_ERROR (reading it clears it). The value is cut to
/// `*len` bytes, which then holds the length copied.
fn kernel_getsockopt(ctx: &CpuContext, fd: i32, level: i32, option: i32, value: u64, len_addr: u64) -> Result<(), i32> {
    let s = socket_ref(fd)?;
    if level != SOL_SOCKET && level != IPPROTO_TCP && level != IPPROTO_IPV6 && level != IPPROTO_IP {
        return Err(ENOPROTOOPT_);
    }
    if len_addr == 0 {
        return Err(EINVAL_);
    }
    let host = s.host.fd;
    let host_int = |l: i32, o: i32| get_int_opt(host, l, o).map_err(map_errno);
    let bytes: Vec<u8> = match (level, option) {
        (IPPROTO_IP, _) => {
            let ipv4 = s.family == ORBIS_AF_INET;
            let datagram = s.ty == SOCK_DGRAM_K;
            let opts = ksockets().get(&fd).map(|k| k.opts.clone()).ok_or(EBADF_)?;
            let capacity = ctx.read_u32(len_addr).unwrap_or(0);
            get_ip_option(&opts, ipv4, datagram, option, capacity)?
        }
        (IPPROTO_TCP, _) if !s.is_tcp() => return Err(EINVAL_),
        (IPPROTO_TCP, _) => {
            let opts = ksockets().get(&fd).map(|k| k.opts.clone()).ok_or(EBADF_)?;
            get_tcp_state_option(host, &opts, option)?.to_le_bytes().to_vec()
        }
        (IPPROTO_IPV6, IPV6_V6ONLY) if s.family == ORBIS_AF_INET6 => {
            ((host_int(libc::IPPROTO_IPV6, libc::IPV6_V6ONLY)? != 0) as i32).to_le_bytes().to_vec()
        }
        (SOL_SOCKET, SO_LINGER) => {
            let mut l = libc::linger { l_onoff: 0, l_linger: 0 };
            let mut n = std::mem::size_of::<libc::linger>() as libc::socklen_t;
            // SAFETY: one linger of its own size.
            if unsafe { libc::getsockopt(host, libc::SOL_SOCKET, libc::SO_LINGER, &mut l as *mut _ as *mut libc::c_void, &mut n) } < 0 {
                return Err(map_errno(last_errno()));
            }
            let mut b = (if l.l_onoff != 0 { SO_LINGER } else { 0 }).to_le_bytes().to_vec();
            b.extend_from_slice(&l.l_linger.to_le_bytes());
            b
        }
        (SOL_SOCKET, SO_SNDTIMEO | SO_RCVTIMEO) => {
            let timeout = {
                let socks = ksockets();
                let k = socks.get(&fd).ok_or(EBADF_)?;
                if option == SO_RCVTIMEO { k.receive_timeout } else { k.send_timeout }
            };
            let (sec, usec) = timeval_of(timeout);
            let mut b = sec.to_le_bytes().to_vec();
            b.extend_from_slice(&usec.to_le_bytes());
            b
        }
        (SOL_SOCKET, _) => {
            let flag = FLAG_OPTIONS.iter().find(|(o, _)| *o == option).map(|&(_, n)| n);
            let stored = |f: fn(&KernelSocket) -> i32| ksockets().get(&fd).map(f).ok_or(EBADF_);
            let v = match option {
                _ if flag.is_some() => {
                    if host_int(libc::SOL_SOCKET, flag.unwrap())? != 0 { option } else { 0 }
                }
                SO_ACCEPTCONN => {
                    if host_int(libc::SOL_SOCKET, libc::SO_ACCEPTCONN)? != 0 { SO_ACCEPTCONN } else { 0 }
                }
                SO_TYPE => s.ty,
                SO_NOSIGPIPE => {
                    if stored(|k| k.no_sigpipe as i32)? != 0 { SO_NOSIGPIPE } else { 0 }
                }
                SO_NBIO => stored(|k| k.nonblocking as i32)?,
                SO_ERROR => map_errno(host_int(libc::SOL_SOCKET, libc::SO_ERROR)?),
                SO_SNDBUF => match stored(|k| k.send_buffer)? {
                    0 => host_int(libc::SOL_SOCKET, libc::SO_SNDBUF)?,
                    v => v,
                },
                SO_RCVBUF => match stored(|k| k.receive_buffer)? {
                    0 => host_int(libc::SOL_SOCKET, libc::SO_RCVBUF)?,
                    v => v,
                },
                SO_SNDLOWAT => stored(|k| k.send_low_water)?,
                SO_RCVLOWAT => stored(|k| k.receive_low_water)?,
                // The kept flags come back from the console's own state.
                _ if kept_or_write_only(option) && option & KEPT_FLAG_OPTIONS != 0 => stored(|k| k.opts.kept_flags)? & option,
                _ => return Err(ENOPROTOOPT_),
            };
            v.to_le_bytes().to_vec()
        }
        _ => return Err(ENOPROTOOPT_),
    };
    let capacity = ctx.read_u32(len_addr).ok_or(EFAULT_)?;
    let n = capacity.min(bytes.len() as u32);
    if n != 0 && value == 0 {
        return Err(EINVAL_);
    }
    if (n != 0 && !ctx.write_bytes(value, &bytes[..n as usize])) || !ctx.write_u32(len_addr, n) {
        return Err(EFAULT_);
    }
    Ok(())
}

/// sendto(fd, buf, len, flags, addr, addrlen)
fn kernel_sendto(ctx: &CpuContext) -> HleResult {
    let r = send_core(ctx, ctx.rdi() as i32, ctx.rsi(), ctx.rdx(), ctx.rcx() as i32, ctx.r8(), ctx.r9() as u32);
    posix_socket_result(ctx, r)
}

/// recvfrom(fd, buf, len, flags, addr, addrlen*)
fn kernel_recvfrom(ctx: &CpuContext) -> HleResult {
    let r = receive_core(ctx, ctx.rdi() as i32, ctx.rsi(), ctx.rdx(), ctx.rcx() as i32, ctx.r8(), ctx.r9());
    posix_socket_result(ctx, r)
}

/// bzero(p, n)
fn cb_bzero(ctx: &mut CpuContext) -> HleResult {
    let addr = ctx.rdi();
    let len = ctx.rsi() as i32;
    if len > 0 && addr != 0 && !ctx.write_bytes(addr, &vec![0u8; len as usize]) {
        return KERNEL_EFAULT;
    }
    0
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("Nlev7Lg8k3A", "sceNetInit", cb_net_init),
    ("cTGkc6-TBlI", "sceNetTerm", cb_net_term),
    ("6Oc0bLsIYe0", "sceNetGetMacAddress", cb_get_mac_address),
    ("v6M4txecCuo", "sceNetEtherNtostr", cb_ether_ntostr),
    ("Q4qBuN-c0ZM", "sceNetSocket", cb_socket),
    ("45ggEzakPJQ", "sceNetSocketClose", cb_socket_close),
    ("9wO9XrMsNhc", "sceNetRecv", cb_recv),
    ("304ooNZxWDY", "sceNetRecvfrom", cb_recvfrom),
    ("2mKX2Spso7I", "sceNetSetsockopt", cb_setsockopt),
    ("bErx49PgxyY", "sceNetBind", cb_bind),
    ("kOj1HiAGE54", "sceNetListen", cb_listen),
    ("PIWqhn9oSxc", "sceNetAccept", cb_accept),
    ("HQOwnfMGipQ", "sceNetErrnoLoc", cb_errno_loc),
    ("dgJBaeJnGpo", "sceNetPoolCreate", cb_pool_create),
    ("K7RlrTkI-mw", "sceNetPoolDestroy", cb_pool_destroy),
    ("rMyh97BU5pY", "sceNetGetMemoryPoolStats", cb_get_memory_pool_stats),
    ("9T2pDF2Ryqg", "sceNetHtonl", cb_htonl),
    ("iWQWrwiSt8A", "sceNetHtons", cb_htons),
    ("pQGpHYopAIY", "sceNetNtohl", cb_htonl),
    ("Rbvt+5Y2iEw", "sceNetNtohs", cb_htons),
    ("C4UgDHHPvdw", "sceNetResolverCreate", cb_resolver_create),
    ("kJlYH5uMAWI", "sceNetResolverDestroy", cb_resolver_destroy),
    ("AzqoBha7js4", "sceNetResolverAbort", cb_resolver_abort),
    ("J5i3hiLJMPk", "sceNetResolverGetError", cb_resolver_get_error),
    ("Nd91WaWmG2w", "sceNetResolverStartNtoa", cb_resolver_start_ntoa),
    ("8Kcp5d-q1Uo", "sceNetInetPton", cb_net_inet_pton),
    ("Xn2TA2QhxHc", "sceNetInetPtonEx", cb_net_inet_pton_ex),
    ("bjrzRLFali0", "sceNetUsleep", cb_net_usleep),
    ("9vA2aW+CHuA", "sceNetInetNtop", cb_inet_ntop),
    ("OXXX4mUk3uk", "sceNetConnect", cb_connect),
    ("beRjXBn-z+o", "sceNetSend", cb_send),
    ("gvD1greCu0A", "sceNetSendto", cb_sendto),
    ("hoOAofhhRvE", "sceNetGetsockname", cb_getsockname),
    ("xphrZusl78E", "sceNetGetsockopt", cb_getsockopt),
    ("hLuXdjHnhiI", "sceNetGetSockInfo", cb_get_sock_info),
    ("TSM6whtekok", "sceNetShutdown", cb_shutdown),
    ("zJGf8xjFnQE", "sceNetSocketAbort", cb_socket_abort),
    ("SF47kB2MNTo", "sceNetEpollCreate", cb_epoll_create),
    ("Inp1lfL+Jdw", "sceNetEpollDestroy", cb_epoll_destroy),
    ("ZVw46bsasAk", "sceNetEpollControl", cb_epoll_control),
    ("drjIbDbA7UQ", "sceNetEpollWait", cb_epoll_wait),
    ("w21YgGGNtBk", "sceNetEpollAbort", cb_epoll_abort),
    ("Apb4YDxKsRI", "sceNetResolverStartAton", cb_resolver_start_aton),
    ("RCCY01Xd+58", "sceNetResolverStartNtoaMultipleRecords", cb_resolver_start_ntoa_multiple),
    // Same arguments and SceNetResolverInfo as the call without Ex.
    ("sT4nBQKUPqM", "sceNetResolverStartNtoaMultipleRecordsEx", cb_resolver_start_ntoa_multiple),
];

const KERNEL_EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("fFxGkxF2bVo", "setsockopt", cb_posix_setsockopt),
    ("6O8EwYOgH9Y", "getsockopt", cb_posix_getsockopt),
    ("fZOeZIOEmLw", "send", cb_posix_send),
    ("oBr313PppNE", "sendto", cb_posix_sendto),
    ("lUk6wrGXyMw", "recvfrom", cb_posix_recvfrom),
    ("5jRCs2axtr4", "inet_ntop", kmsg::cb_inet_ntop_errno),
    ("TU-d9PfIHPM", "socket", cb_kernel_socket),
    ("pG70GT5yRo4", "__sys_socketex", cb_kernel_socketex),
    ("XVL8So3QJUk", "connect", cb_kernel_connect),
    ("KuOmgKoqCdY", "bind", cb_kernel_bind),
    ("RenI1lL1WFk", "getsockname", cb_kernel_getsockname),
    ("9oiX1kyeedA", "bzero", cb_bzero),
    ("4n51s0zEf0c", "inet_pton", kmsg::cb_inet_pton_errno),
    ("jogUIsOV3-U", "htons", cb_htons),
    ("K1S8oc61xiM", "htonl", cb_htonl),
    ("jct5WjixLgk", "ntohl", cb_htonl),
    ("oLFi+HuZ7hY", "ntohs", cb_htons),
];

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, cb) in EXPORTS.iter().chain(scenet_ext::EXPORTS) {
        reg.register_fn(nid, name, LIB, cb);
    }
    // The sceSp* wrappers of libSceSpSysCallWrapper forward to these.
    const SP_SYSCALL_EXPORTS: &[(&str, &str, HleCallback)] = &[
        ("GNfPC72c3fs", "sceSpNetConnect", cb_connect),
        ("7pl2UBkJIRk", "sceSpNetEpollAbort", cb_epoll_abort),
        ("loIvEsvUejI", "sceSpNetEpollControl", cb_epoll_control),
        ("L25oH9QaTaQ", "sceSpNetEpollCreate", cb_epoll_create),
        ("EGt+sFM8+WI", "sceSpNetEpollDestroy", cb_epoll_destroy),
        ("uaXURZ5NMGw", "sceSpNetEpollWait", cb_epoll_wait),
        ("Uv+1RNg62iU", "sceSpNetErrnoLoc", cb_errno_loc),
        ("sTqVHpVwnyE", "sceSpNetPoolCreate", cb_pool_create),
        ("yrOv0czHCp4", "sceSpNetPoolDestroy", cb_pool_destroy),
        ("p8VOOvRtBwI", "sceSpNetRecv", cb_recv),
        ("k1OoMsQ7tQ8", "sceSpNetSend", cb_send),
        ("81ZjjD6MqDk", "sceSpNetResolverAbort", cb_resolver_abort),
        ("y6AEzLH6ZXk", "sceSpNetResolverCreate", cb_resolver_create),
        ("2mcSJjH5VLc", "sceSpNetResolverDestroy", cb_resolver_destroy),
        ("DVAUSbVTZnU", "sceSpNetResolverGetError", cb_resolver_get_error),
        ("WwtboQmJ37U", "sceSpNetResolverStartNtoa", cb_resolver_start_ntoa),
        ("akGUA-sxyYk", "sceSpNetSetsockopt", cb_setsockopt),
        ("kyKiXG-dCyc", "sceSpNetSocket", cb_socket),
        ("2-bYnN1qoQM", "sceSpNetSocketClose", cb_socket_close),
    ];
    for &(nid, name, cb) in SP_SYSCALL_EXPORTS {
        reg.register_fn(nid, name, "libSceSpSysCallWrapper", cb);
    }
    for &(nid, name, cb) in KERNEL_EXPORTS.iter().chain(posix_ext::EXPORTS).chain(kmsg::KMSG_EXPORTS) {
        reg.register_fn(nid, name, LIB_KERNEL, cb);
    }
    for &(nid, name, cb) in KERNEL_EXPORTS.iter().chain(posix_ext::EXPORTS).chain(kmsg::KMSG_EXPORTS) {
        reg.register_fn(nid, name, LIB_KERNEL, cb);
    }
    for &(nid, name, cb) in posix_ext::POSIX_WEBKIT_EXPORTS {
        reg.register_fn(nid, name, "libScePosixForWebKit", cb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CpuRegister::*;

    const NET_ERROR_INVALID_ARGUMENT: i32 = net_error(EINVAL_);
    const NET_ERROR_BAD_FD: i32 = net_error(EBADF_);

    /// Socket ids are the lowest free ones, so tests that close sockets
    /// (and check a second close fails) must not interleave.
    pub(super) fn socket_ids() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[test]
    fn nids_match_names() {
        for &(nid, name, _) in EXPORTS.iter().chain(KERNEL_EXPORTS) {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    #[test]
    fn sockaddr_helpers() {
        let sa = guest_sockaddr_in([127, 0, 0, 1], 8080);
        assert_eq!(sa[0], 16);
        assert_eq!(sa[1], 2);
        assert_eq!(&sa[2..4], &[0x1F, 0x90]);
        assert_eq!(&sa[4..8], &[127, 0, 0, 1]);
        assert_eq!(parse_guest_sockaddr_in(&sa), Some((Ipv4Addr::LOCALHOST, 8080)));
        let mut bad = sa;
        bad[1] = 28;
        assert_eq!(parse_guest_sockaddr_in(&bad), None);
        assert_eq!(map_to_ipv4("::ffff:10.1.2.3".parse().unwrap()), [10, 1, 2, 3]);
    }

    #[test]
    fn address_literals() {
        assert_eq!(parse_ipv4_literal("192.168.0.1"), Some(Ipv4Addr::new(192, 168, 0, 1)));
        assert_eq!(parse_ipv4_literal("127.1"), Some(Ipv4Addr::new(127, 0, 0, 1)));
        assert_eq!(parse_ipv4_literal("0x7f.0.0.1"), Some(Ipv4Addr::new(127, 0, 0, 1)));
        assert_eq!(parse_ipv4_literal("256.0.0.1"), None);
        assert_eq!(parse_ipv4_literal("example.com"), None);
        assert_eq!(parse_ip_literal("::1"), Some(IpAddr::V6(Ipv6Addr::LOCALHOST)));
        assert_eq!(format_address(2, &[8, 8, 4, 4]).as_deref(), Some("8.8.4.4"));
        assert_eq!(format_address(28, &Ipv6Addr::LOCALHOST.octets()).as_deref(), Some("::1"));
        assert_eq!(format_address(1, &[0; 16]), None);
    }

    #[test]
    fn kernel_udp_loopback() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5C04_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let set = |ctx: &mut CpuContext, args: &[u64]| {
            for (r, v) in [Rdi, Rsi, Rdx, Rcx, R8, R9].iter().zip(args) {
                ctx.set(*r, *v);
            }
        };
        set(&mut ctx, &[2, 2, 0]);
        let fd = cb_kernel_socket(&mut ctx);
        assert!(fd > 2);
        let fd = fd as u64;
        let sa = BASE;
        ctx.write_bytes(sa, &guest_sockaddr_in([127, 0, 0, 1], 0));
        set(&mut ctx, &[fd, sa, 16]);
        assert_eq!(cb_kernel_bind(&mut ctx), 0);
        let (name, len) = (BASE + 0x40, BASE + 0x60);
        ctx.write_u32(len, 16);
        set(&mut ctx, &[fd, name, len]);
        assert_eq!(cb_kernel_getsockname(&mut ctx), 0);
        let local = ctx.read_bytes(name, 16).unwrap();
        let (ip, port) = parse_guest_sockaddr_in(&local).unwrap();
        assert_eq!(ip, Ipv4Addr::LOCALHOST);
        assert_ne!(port, 0);

        ctx.write_bytes(BASE + 0x100, b"ping");
        set(&mut ctx, &[fd, BASE + 0x100, 4, 0, name, 16]);
        assert_eq!(cb_posix_sendto(&mut ctx), 4);
        ctx.write_u32(len, 16);
        set(&mut ctx, &[fd, BASE + 0x200, 64, MSG_DONTWAIT as u64, BASE + 0x80, len]);
        assert_eq!(cb_posix_recvfrom(&mut ctx), 4);
        assert_eq!(ctx.read_bytes(BASE + 0x200, 4).unwrap(), b"ping".to_vec());
        assert_eq!(parse_guest_sockaddr_in(&ctx.read_bytes(BASE + 0x80, 16).unwrap()), Some((ip, port)));

        // Non-loopback destinations are refused while offline.
        if std::env::var("RPS5E_NET_REDIRECT").is_err() {
            ctx.write_bytes(BASE + 0xA0, &guest_sockaddr_in([8, 8, 8, 8], 53));
            set(&mut ctx, &[fd, BASE + 0x100, 4, 0, BASE + 0xA0, 16]);
            assert_eq!(cb_posix_sendto(&mut ctx), -1);
        }
        // SO_REUSEADDR round trip through the POSIX option calls.
        ctx.write_u32(BASE + 0x300, 1);
        set(&mut ctx, &[fd, SOL_SOCKET as u64, SO_REUSEADDR as u64, BASE + 0x300, 4]);
        assert_eq!(cb_posix_setsockopt(&mut ctx), 0);
        ctx.write_u32(BASE + 0x308, 4);
        set(&mut ctx, &[fd, SOL_SOCKET as u64, SO_REUSEADDR as u64, BASE + 0x304, BASE + 0x308]);
        assert_eq!(cb_posix_getsockopt(&mut ctx), 0);
        assert_eq!(ctx.read_u32(BASE + 0x304), Some(SO_REUSEADDR as u32), "FreeBSD reports the option bit");
        assert!(kernel_socket_close(fd as i32));
        assert!(!is_kernel_socket(fd as i32));
    }

    #[test]
    fn byte_order_and_pton_in_guest_memory() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5C00_0000_0000;
        vm.map(BASE, 0x10000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.set(Rdi, 0x1234);
        assert_eq!(cb_htons(&mut ctx), 0x3412);
        ctx.set(Rdi, 0x1122_3344);
        assert_eq!(cb_htonl(&mut ctx), 0x4433_2211);

        ctx.write_bytes(BASE, b"10.20.30.40\0");
        ctx.set(Rdi, 2);
        ctx.set(Rsi, BASE);
        ctx.set(Rdx, BASE + 0x100);
        assert_eq!(kmsg::cb_inet_pton_errno(&mut ctx), 1);
        assert_eq!(ctx.read_bytes(BASE + 0x100, 4).unwrap(), vec![10, 20, 30, 40]);
        assert_eq!(cb_net_inet_pton(&mut ctx), 1);

        // inet_ntop round trip, then a destination that is one byte short.
        ctx.set(Rdi, 2);
        ctx.set(Rsi, BASE + 0x100);
        ctx.set(Rdx, BASE + 0x200);
        ctx.set(Rcx, 16);
        assert_eq!(cb_inet_ntop(&mut ctx), (BASE + 0x200) as i64);
        assert_eq!(ctx.read_cstring(BASE + 0x200, 32).unwrap(), "10.20.30.40");
        ctx.set(Rcx, 11);
        assert_eq!(cb_inet_ntop(&mut ctx), 0);
    }
    /// Exact and larger buffers receive the text and terminator only;
    /// short buffers and NULL arguments fail without writing.
    #[test]
    fn ether_address_formatting() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5C60_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let (addr, text) = (BASE, BASE + 0x100);
        let set = |ctx: &mut CpuContext, args: [u64; 3]| {
            for (r, v) in [Rdi, Rsi, Rdx].iter().zip(args) {
                ctx.set(*r, v);
            }
        };
        for (mac, expected) in [([0u8; 6], "00:00:00:00:00:00"), ([0x01, 0x23, 0x45, 0xab, 0xcd, 0xef], "01:23:45:ab:cd:ef")] {
            ctx.write_bytes(addr, &mac);
            for size in [18u64, 127] {
                ctx.write_bytes(text, &[b'!'; 128]);
                set(&mut ctx, [addr, text, size]);
                assert_eq!(cb_ether_ntostr(&mut ctx), 0);
                let out = ctx.read_bytes(text, 128).unwrap();
                assert_eq!(&out[..17], expected.as_bytes());
                assert_eq!(out[17], 0);
                assert_eq!(out[18], b'!');
            }
        }
        ctx.write_bytes(addr, &[0u8; 6]);
        ctx.write_bytes(text, &[b'!'; 18]);
        let invalid = NET_ERROR_INVALID_ARGUMENT as i64;
        set(&mut ctx, [addr, text, 17]);
        assert_eq!(cb_ether_ntostr(&mut ctx), invalid);
        assert_eq!(ctx.read_bytes(text, 18).unwrap(), vec![b'!'; 18]);
        set(&mut ctx, [0, text, 18]);
        assert_eq!(cb_ether_ntostr(&mut ctx), invalid);
        set(&mut ctx, [addr, 0, 18]);
        assert_eq!(cb_ether_ntostr(&mut ctx), invalid);
        assert_eq!(ctx.read_bytes(text, 18).unwrap(), vec![b'!'; 18]);
        let _ = vm.unmap(BASE, 0x1000);
    }

    #[test]
    fn sce_net_udp_epoll_and_offline_policy() {
        let _ids = socket_ids();
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5C06_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let set = |ctx: &mut CpuContext, args: &[u64]| {
            for (r, v) in [Rdi, Rsi, Rdx, Rcx, R8, R9].iter().zip(args) {
                ctx.set(*r, *v);
            }
        };
        ctx.write_bytes(BASE, b"udp\0");
        set(&mut ctx, &[BASE, 2, 2, 0]);
        let s = cb_socket(&mut ctx);
        assert!((SOCKET_FD_MIN..SOCKET_FD_MAX).contains(&(s as i32)));
        let s = s as u64;
        ctx.write_bytes(BASE + 0x10, &guest_sockaddr_in([127, 0, 0, 1], 0));
        set(&mut ctx, &[s, BASE + 0x10, 16]);
        assert_eq!(cb_bind(&mut ctx), 0);
        ctx.write_u32(BASE + 0x30, 16);
        set(&mut ctx, &[s, BASE + 0x20, BASE + 0x30]);
        assert_eq!(cb_getsockname(&mut ctx), 0);
        let (ip, port) = parse_guest_sockaddr_in(&ctx.read_bytes(BASE + 0x20, 16).unwrap()).unwrap();
        assert_eq!(ip, Ipv4Addr::LOCALHOST);
        assert_ne!(port, 0);

        // Epoll: nothing readable yet -> honest timeout.
        ctx.write_bytes(BASE + 0x40, b"ep\0");
        set(&mut ctx, &[BASE + 0x40, 0]);
        let ep = cb_epoll_create(&mut ctx) as u64;
        ctx.write_u32(BASE + 0x50, NET_EPOLLIN);
        ctx.write_u64(BASE + 0x60, 0xABCD);
        set(&mut ctx, &[ep, NET_EPOLL_CTL_ADD as u64, s, BASE + 0x50]);
        assert_eq!(cb_epoll_control(&mut ctx), 0);
        assert_eq!(cb_epoll_control(&mut ctx), code(0x8041_0111) as i64);
        set(&mut ctx, &[ep, BASE + 0x100, 4, 20_000]);
        let t = std::time::Instant::now();
        assert_eq!(cb_epoll_wait(&mut ctx), 0);
        assert!(t.elapsed() >= std::time::Duration::from_millis(15));

        // Loopback datagram to ourselves makes the socket readable.
        ctx.write_bytes(BASE + 0x200, b"hi");
        set(&mut ctx, &[s, BASE + 0x200, 2, 0, BASE + 0x20, 16]);
        assert_eq!(cb_sendto(&mut ctx), 2);
        set(&mut ctx, &[ep, BASE + 0x100, 4, 1_000_000]);
        assert_eq!(cb_epoll_wait(&mut ctx), 1);
        assert_eq!(ctx.read_u32(BASE + 0x100), Some(NET_EPOLLIN));
        assert_eq!(ctx.read_u64(BASE + 0x108), Some(s));
        assert_eq!(ctx.read_u64(BASE + 0x110), Some(0xABCD));

        // Non-loopback destinations are unreachable offline.
        if std::env::var("RPS5E_NET_REDIRECT").is_err() {
            ctx.write_bytes(BASE + 0x300, &guest_sockaddr_in([8, 8, 8, 8], 53));
            set(&mut ctx, &[s, BASE + 0x200, 2, 0, BASE + 0x300, 16]);
            assert_eq!(cb_sendto(&mut ctx), code(0x8041_0133) as i64);
            set(&mut ctx, &[s, BASE + 0x300, 16]);
            assert_eq!(cb_connect(&mut ctx), code(0x8041_0133) as i64);
        }

        // Socket info and options come from the host socket.
        set(&mut ctx, &[s, BASE + 0x400, 1, 0]);
        assert_eq!(cb_get_sock_info(&mut ctx), 1);
        assert_eq!(ctx.read_cstring(BASE + 0x400, 32).unwrap(), "udp");
        // The flags (SELF, and NONBLOCK when set) and the buffer sizes.
        assert_eq!(ctx.read_u32(BASE + 0x400 + 0x48).map(|f| f & 1), Some(1));
        assert!(ctx.read_u32(BASE + 0x400 + 0x64).unwrap() > 0);
        assert!(ctx.read_u32(BASE + 0x400 + 0x68).unwrap() > 0);
        // A flag other than TIME_WAIT, no buffer for one socket, a count
        // under 1: EINVAL; no buffer for every socket: the count.
        for bad in [[s, BASE + 0x400, 1, 1], [s, 0, 1, 0], [s, BASE + 0x400, 0, 0]] {
            set(&mut ctx, &bad);
            assert_eq!(cb_get_sock_info(&mut ctx), code(0x8041_0116) as i64, "{bad:?}");
        }
        set(&mut ctx, &[u32::MAX as u64, 0, 0, 0x40]);
        assert!(cb_get_sock_info(&mut ctx) >= 1);
        ctx.write_u32(BASE + 0x48, 4);
        set(&mut ctx, &[s, SOL_SOCKET as u64, SO_TYPE as u64, BASE + 0x44, BASE + 0x48]);
        assert_eq!(cb_getsockopt(&mut ctx), 0);
        assert_eq!(ctx.read_u32(BASE + 0x44), Some(2));
        // The flag options read back as their own bit, cut to the length;
        // a datagram socket is not listening.
        for (option, on) in [(SO_BROADCAST, 7u32), (SO_OOBINLINE, 1), (SO_KEEPALIVE, 256), (SO_OOBINLINE, 0)] {
            ctx.write_u32(BASE + 0x40, on);
            set(&mut ctx, &[s, SOL_SOCKET as u64, option as u64, BASE + 0x40, 4]);
            assert_eq!(cb_setsockopt(&mut ctx), 0, "{option:#x}");
            for capacity in [4u32, 1] {
                ctx.write_u32(BASE + 0x44, 0xa5a5_a5a5);
                ctx.write_u32(BASE + 0x48, capacity);
                set(&mut ctx, &[s, SOL_SOCKET as u64, option as u64, BASE + 0x44, BASE + 0x48]);
                assert_eq!(cb_getsockopt(&mut ctx), 0);
                let want = if on != 0 { option as u32 } else { 0 };
                let mask = if capacity == 4 { u32::MAX } else { 0xff };
                assert_eq!(ctx.read_u32(BASE + 0x44), Some((want & mask) | (0xa5a5_a5a5 & !mask)), "{option:#x}");
                assert_eq!(ctx.read_u32(BASE + 0x48), Some(capacity));
            }
        }
        ctx.write_u32(BASE + 0x48, 4);
        set(&mut ctx, &[s, SOL_SOCKET as u64, SO_ACCEPTCONN as u64, BASE + 0x44, BASE + 0x48]);
        assert_eq!(cb_getsockopt(&mut ctx), 0);
        assert_eq!(ctx.read_u32(BASE + 0x44), Some(0));

        set(&mut ctx, &[ep, NET_EPOLL_CTL_DEL as u64, s, 0]);
        assert_eq!(cb_epoll_control(&mut ctx), 0);
        set(&mut ctx, &[ep]);
        assert_eq!(cb_epoll_destroy(&mut ctx), 0);
        assert_eq!(cb_epoll_destroy(&mut ctx), NET_ERROR_BAD_FD as i64);
        set(&mut ctx, &[s]);
        assert_eq!(cb_socket_close(&mut ctx), 0);
        set(&mut ctx, &[s, BASE + 0x200, 2, 0]);
        assert_eq!(cb_send(&mut ctx), NET_ERROR_BAD_FD as i64);
    }

    #[test]
    fn empty_epoll_waits_abort_and_destroy() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5C06_0010_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let set = |ctx: &mut CpuContext, args: &[u64]| {
            for (r, v) in [Rdi, Rsi, Rdx, Rcx].iter().zip(args) {
                ctx.set(*r, *v);
            }
        };
        let aborted = net_error(ECONNABORTED_) as i64;
        ctx.write_bytes(BASE, b"empty-epoll\0");
        set(&mut ctx, &[BASE, 0]);
        let ep = cb_epoll_create(&mut ctx) as u64;
        let events = BASE + 0x100;
        set(&mut ctx, &[ep, events, 1, 0]);
        assert_eq!(cb_epoll_wait(&mut ctx), 0);
        for _ in 0..3 {
            let t = std::time::Instant::now();
            set(&mut ctx, &[ep, events, 1, 1000]);
            assert_eq!(cb_epoll_wait(&mut ctx), 0);
            assert!(t.elapsed() >= std::time::Duration::from_micros(1000));
        }
        // An abort fails the next wait once.
        set(&mut ctx, &[ep, 0]);
        assert_eq!(cb_epoll_abort(&mut ctx), 0);
        set(&mut ctx, &[ep, events, 1, 1000]);
        assert_eq!(cb_epoll_wait(&mut ctx), aborted);
        set(&mut ctx, &[ep, events, 1, 0]);
        assert_eq!(cb_epoll_wait(&mut ctx), 0);
        set(&mut ctx, &[ep]);
        assert_eq!(cb_epoll_destroy(&mut ctx), 0);
        set(&mut ctx, &[ep, 0]);
        assert_eq!(cb_epoll_abort(&mut ctx), NET_ERROR_BAD_FD as i64);

        // A wait without timeout ends on abort (ECONNABORTED) or on
        // destruction of the epoll (EBADF).
        for abort in [true, false] {
            ctx.write_bytes(BASE, b"waiting-epoll\0");
            set(&mut ctx, &[BASE, 0]);
            let ep = cb_epoll_create(&mut ctx) as u64;
            let waiter = std::thread::spawn(move || {
                let vm = ps5vm::VirtualMemory::new();
                let mut ctx = CpuContext::new(&vm);
                for (r, v) in [(Rdi, ep), (Rsi, events), (Rdx, 1), (Rcx, u32::MAX as u64)] {
                    ctx.set(r, v);
                }
                cb_epoll_wait(&mut ctx)
            });
            std::thread::sleep(std::time::Duration::from_millis(20));
            set(&mut ctx, &[ep, 0]);
            assert_eq!(if abort { cb_epoll_abort(&mut ctx) } else { cb_epoll_destroy(&mut ctx) }, 0);
            let expected = if abort { aborted } else { NET_ERROR_BAD_FD as i64 };
            assert_eq!(waiter.join().unwrap(), expected);
            if abort {
                assert_eq!(cb_epoll_destroy(&mut ctx), 0);
            }
        }
    }

    fn args(ctx: &mut CpuContext, args: &[u64]) {
        for (r, v) in [Rdi, Rsi, Rdx, Rcx, R8, R9].iter().zip(args) {
            ctx.set(*r, *v);
        }
    }

    /// Pools take ids 1..31; resolvers need a live pool and a name; every
    /// failure also stores its errno.
    #[test]
    fn pools_resolvers_and_errno() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5D00_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.write_bytes(BASE, b"pool\0");
        args(&mut ctx, &[BASE, 0x4000, 0]);
        let pool = cb_pool_create(&mut ctx);
        assert!((1..POOLS_MAX as i64).contains(&pool));
        let errno = cb_errno_loc(&mut ctx) as u64;
        assert_ne!(errno, 0);

        args(&mut ctx, &[0, pool as u64, 0]);
        assert_eq!(cb_resolver_create(&mut ctx), net_error(EINVAL_) as i64);
        assert_eq!(ctx.read_u32(errno), Some(EINVAL_ as u32));
        args(&mut ctx, &[BASE, pool as u64, 1]);
        assert_eq!(cb_resolver_create(&mut ctx), net_error(EINVAL_) as i64);
        args(&mut ctx, &[BASE, 0x7777, 0]);
        assert_eq!(cb_resolver_create(&mut ctx), NET_ERROR_RESOLVER_ENOSPACE as i64);
        assert_eq!(ctx.read_u32(errno), Some(0xDE));
        args(&mut ctx, &[BASE, pool as u64, 0]);
        let rid = cb_resolver_create(&mut ctx);
        assert!((0..RESOLVERS_MAX as i64).contains(&rid));

        // Argument checks come before the resolver id check.
        ctx.write_bytes(BASE + 0x100, b"192.168.1.20\0");
        args(&mut ctx, &[0x7777, BASE + 0x100, 0, 0, 0, 0]);
        assert_eq!(cb_resolver_start_ntoa(&mut ctx), net_error(EINVAL_) as i64);
        args(&mut ctx, &[0x7777, BASE + 0x100, BASE + 0x200, 0, 0, 0]);
        assert_eq!(cb_resolver_start_ntoa(&mut ctx), NET_ERROR_BAD_FD as i64);
        args(&mut ctx, &[rid as u64, BASE + 0x100, BASE + 0x200, 0, 0, 0x2]);
        assert_eq!(cb_resolver_start_ntoa(&mut ctx), net_error(EINVAL_) as i64);
        for (timeout, retry, flags) in [(-1i32, 0i32, 0u32), (0, -1, 0), (5_000_000, 1, 0x2_0000), (0, 0, 0x1_0003), (0, 0, 0x8000_0000)] {
            args(&mut ctx, &[0x7777, BASE + 0x100, BASE + 0x200, timeout as u32 as u64, retry as u32 as u64, flags as u64]);
            assert_eq!(cb_resolver_start_ntoa(&mut ctx), net_error(EINVAL_) as i64, "{timeout} {retry} {flags:#x}");
        }
        for flags in [1u64, 0x1_0000, 0x1_0001] {
            args(&mut ctx, &[0x7777, BASE + 0x100, BASE + 0x200, 0, 0, flags]);
            assert_eq!(cb_resolver_start_ntoa(&mut ctx), NET_ERROR_BAD_FD as i64, "{flags:#x}");
        }
        args(&mut ctx, &[rid as u64, BASE + 0x100, BASE + 0x200, 0, 0, 0]);
        assert_eq!(cb_resolver_start_ntoa(&mut ctx), 0);
        assert_eq!(ctx.read_bytes(BASE + 0x200, 4).unwrap(), vec![192, 168, 1, 20]);
        // Only strict dotted quads are literals.
        ctx.write_bytes(BASE + 0x100, b"127.1\0");
        assert_eq!(cb_resolver_start_ntoa(&mut ctx), NET_ERROR_RESOLVER_NO_DNS as i64);
        assert_eq!(ctx.read_u32(errno), Some(0xE1));
        args(&mut ctx, &[rid as u64, 0]);
        assert_eq!(cb_resolver_abort(&mut ctx), 0);

        args(&mut ctx, &[rid as u64]);
        assert_eq!(cb_resolver_destroy(&mut ctx), 0);
        assert_eq!(cb_resolver_destroy(&mut ctx), NET_ERROR_BAD_FD as i64);

        // Pool statistics: the created size, nothing in use.
        ctx.write_bytes(BASE + 0x300, &[1; 24]);
        args(&mut ctx, &[pool as u64, BASE + 0x300]);
        assert_eq!(cb_get_memory_pool_stats(&mut ctx), 0);
        assert_eq!(ctx.read_u64(BASE + 0x300), Some(0x4000));
        assert_eq!((ctx.read_u64(BASE + 0x308), ctx.read_u64(BASE + 0x310)), (Some(0), Some(0)));
        args(&mut ctx, &[pool as u64, 0]);
        assert_eq!(cb_get_memory_pool_stats(&mut ctx), net_error(EINVAL_) as i64);
        assert_eq!(ctx.read_u32(errno), Some(EINVAL_ as u32));
        args(&mut ctx, &[pool as u64]);
        assert_eq!(cb_pool_destroy(&mut ctx), 0);
        assert_eq!(cb_pool_destroy(&mut ctx), NET_ERROR_BAD_FD as i64);
        assert_eq!(ctx.read_u32(errno), Some(EBADF_ as u32));
        args(&mut ctx, &[pool as u64, BASE + 0x300]);
        assert_eq!(cb_get_memory_pool_stats(&mut ctx), NET_ERROR_BAD_FD as i64);
        assert_eq!(ctx.read_u32(errno), Some(EBADF_ as u32));
        // The pool is checked before the output (measured on a console).
        args(&mut ctx, &[pool as u64, 0]);
        assert_eq!(cb_get_memory_pool_stats(&mut ctx), NET_ERROR_BAD_FD as i64);
        args(&mut ctx, &[12345, 0]);
        assert_eq!(cb_get_memory_pool_stats(&mut ctx), NET_ERROR_BAD_FD as i64);
    }

    /// SceNetResolverInfo: 32-byte records with the family at +16, the
    /// counts at +320/+324, nothing written past 384 bytes or on errors.
    #[test]
    fn resolver_multiple_records_layout() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5E7A_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.write_bytes(BASE, b"pool\0");
        args(&mut ctx, &[BASE, 0x4000, 0]);
        let pool = cb_pool_create(&mut ctx);
        args(&mut ctx, &[BASE, pool as u64, 0]);
        let rid = cb_resolver_create(&mut ctx) as u64;
        let info = BASE + 0x200;
        let errno = cb_errno_loc(&mut ctx) as u64;
        for name in ["sceNetResolverStartNtoaMultipleRecords", "sceNetResolverStartNtoaMultipleRecordsEx"] {
            let cb = EXPORTS.iter().find(|e| e.1 == name).unwrap().2;
            ctx.write_bytes(info, &[0xA5; 512]);
            ctx.write_bytes(BASE + 0x100, b"127.0.0.1\0");
            args(&mut ctx, &[rid, BASE + 0x100, info, 5_000_000, 1, 0]);
            assert_eq!(cb(&mut ctx), 0, "{name}");
            let b = ctx.read_bytes(info, 512).unwrap();
            let i32_at = |o: usize| i32::from_le_bytes(b[o..o + 4].try_into().unwrap());
            assert_eq!(&b[..4], &[127, 0, 0, 1]);
            assert_eq!((i32_at(16), i32_at(320), i32_at(324)), (ORBIS_AF_INET, 1, 1));
            assert!(b[20..320].iter().all(|&v| v == 0) && b[4..16].iter().all(|&v| v == 0));
            assert!(b[328..384].iter().all(|&v| v == 0));
            assert!(b[384..].iter().all(|&v| v == 0xA5), "{name} wrote past the info");
            // IPv6 literals are records but not IPv4 ones.
            ctx.write_bytes(BASE + 0x100, b"::1\0");
            assert_eq!(cb(&mut ctx), 0);
            let b6 = ctx.read_bytes(info, 384).unwrap();
            assert_eq!(b6[15], 1);
            assert_eq!(i32::from_le_bytes(b6[16..20].try_into().unwrap()), ORBIS_AF_INET6);
            assert_eq!((b6[320], b6[324]), (1, 0));

            ctx.write_bytes(BASE + 0x100, b"127.0.0.1\0");
            assert_eq!(cb(&mut ctx), 0);
            let resolved = ctx.read_bytes(info, 512).unwrap();
            args(&mut ctx, &[rid, 0, info, 5_000_000, 1, 0]);
            assert_eq!(cb(&mut ctx), net_error(EINVAL_) as i64);
            assert_eq!(ctx.read_u32(errno), Some(EINVAL_ as u32));
            args(&mut ctx, &[rid, BASE + 0x100, 0, 5_000_000, 1, 0]);
            assert_eq!(cb(&mut ctx), net_error(EINVAL_) as i64);
            ctx.write_bytes(BASE + 0x100, b"guest-sce-net.invalid\0");
            args(&mut ctx, &[rid, BASE + 0x100, info, 5_000_000, 1, 0]);
            assert_eq!(cb(&mut ctx), NET_ERROR_RESOLVER_NO_DNS as i64);
            assert_eq!(ctx.read_bytes(info, 512).unwrap(), resolved, "{name} wrote the info on an error");
            args(&mut ctx, &[0x7777, BASE + 0x100, info, 5_000_000, 1, 0]);
            assert_eq!(cb(&mut ctx), NET_ERROR_BAD_FD as i64);
            assert_eq!(ctx.read_u32(errno), Some(EBADF_ as u32));
            assert_eq!(ctx.read_bytes(info, 512).unwrap(), resolved);
        }
        args(&mut ctx, &[rid]);
        assert_eq!(cb_resolver_destroy(&mut ctx), 0);
        args(&mut ctx, &[pool as u64]);
        assert_eq!(cb_pool_destroy(&mut ctx), 0);
    }

    /// inet_aton forms (reference GuestSceNet sceNetInetPtonEx cases).
    #[test]
    fn inet_aton_forms() {
        let ip = |s: &str| inet_aton(s.as_bytes()).map(|v| std::net::Ipv4Addr::from(v).to_string());
        assert_eq!(ip("1.2.3").as_deref(), Some("1.2.0.3"));
        assert_eq!(ip("127.1").as_deref(), Some("127.0.0.1"));
        assert_eq!(ip("0989").as_deref(), Some("0.0.3.221"));
        assert_eq!(ip("0x7f.1").as_deref(), Some("127.0.0.1"));
        assert_eq!(ip("010.0.0.1").as_deref(), Some("8.0.0.1"));
        assert_eq!(ip("1.4294967296").as_deref(), Some("1.0.0.0"));
        assert_eq!(ip("1.2.3.4\tz").as_deref(), Some("1.2.3.4"));
        for bad in ["08", "", "1.2.3.4.5", "256.1", "1.2.3.256", "1.16777216", "a", "1.2.3.4x", " 1"] {
            assert_eq!(ip(bad), None, "{bad}");
        }
    }

    #[test]
    fn pton_mac_and_sockaddr_rules() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5D01_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.write_bytes(BASE, b"127.1\0");
        args(&mut ctx, &[2, BASE, BASE + 0x100]);
        assert_eq!(cb_net_inet_pton(&mut ctx), 0);
        args(&mut ctx, &[1, BASE, BASE + 0x100]);
        assert_eq!(cb_net_inet_pton(&mut ctx), net_error(EAFNOSUPPORT_) as i64);
        ctx.write_bytes(BASE, b"fe80::1\0");
        args(&mut ctx, &[28, BASE, BASE + 0x100]);
        assert_eq!(cb_net_inet_pton(&mut ctx), 1);
        assert_eq!(ctx.read_bytes(BASE + 0x100, 2).unwrap(), vec![0xFE, 0x80]);
        // IPv6 text takes FreeBSD's inet_pton forms: an embedded dotted
        // quad, but no leading zero in its octets (the target is kept).
        for (text, tail) in [(&b"::ffff:0.0.0.1\0"[..], [0u8, 0, 0, 1]), (b"::ffff:0:102:304\0", [1, 2, 3, 4]), (b"fe80::5efe:102:304\0", [1, 2, 3, 4])] {
            ctx.write_bytes(BASE, text);
            assert_eq!(cb_net_inet_pton(&mut ctx), 1, "{text:?}");
            assert_eq!(ctx.read_bytes(BASE + 0x10C, 4).unwrap(), tail.to_vec());
        }
        for bad in [&b"::01.2.3.4\0"[..], b"::ffff:01.2.3.4\0"] {
            ctx.write_bytes(BASE + 0x100, &[0xa5; 16]);
            ctx.write_bytes(BASE, bad);
            assert_eq!(cb_net_inet_pton(&mut ctx), 0, "{bad:?}");
            assert_eq!(ctx.read_bytes(BASE + 0x100, 16).unwrap(), vec![0xa5; 16]);
        }
        // sceNetInetPtonEx: flags 0 is the plain call.
        ctx.write_bytes(BASE, b"1.2.3.4\0");
        args(&mut ctx, &[2, BASE, BASE + 0x100, 0]);
        assert_eq!(cb_net_inet_pton_ex(&mut ctx), 1);
        assert_eq!(ctx.read_bytes(BASE + 0x100, 4).unwrap(), vec![1, 2, 3, 4]);
        // The flags are not looked at; IPv4 text is read like inet_aton.
        args(&mut ctx, &[2, BASE, BASE + 0x100, 1]);
        assert_eq!(cb_net_inet_pton_ex(&mut ctx), 1);
        ctx.write_bytes(BASE, b"127.1\0");
        assert_eq!(cb_net_inet_pton_ex(&mut ctx), 1);
        assert_eq!(ctx.read_bytes(BASE + 0x100, 4).unwrap(), vec![127, 0, 0, 1]);
        ctx.write_bytes(BASE + 0x100, &[9; 4]);
        ctx.write_bytes(BASE, b"08\0");
        assert_eq!(cb_net_inet_pton_ex(&mut ctx), 0);
        assert_eq!(ctx.read_bytes(BASE + 0x100, 4).unwrap(), vec![9; 4]);
        args(&mut ctx, &[2, 0, BASE + 0x100, 0]);
        assert_eq!(cb_net_inet_pton_ex(&mut ctx), net_error(EINVAL_) as i64);
        // sceNetInetPton: IPv4 text ends at whitespace.
        ctx.write_bytes(BASE, b"1.2.3.4 x\0");
        args(&mut ctx, &[2, BASE, BASE + 0x100]);
        assert_eq!(cb_net_inet_pton(&mut ctx), 1);
        assert_eq!(ctx.read_bytes(BASE + 0x100, 4).unwrap(), vec![1, 2, 3, 4]);
        for bad in [&b" 1.2.3.4\0"[..], b"1.2.3 \0", b"1.2.3.4\x01\0"] {
            ctx.write_bytes(BASE, bad);
            assert_eq!(cb_net_inet_pton(&mut ctx), 0, "{bad:?}");
        }
        // sceNetUsleep: a negative time is EINVAL.
        args(&mut ctx, &[u32::MAX as u64]);
        assert_eq!(cb_net_usleep(&mut ctx), net_error(EINVAL_) as i64);
        args(&mut ctx, &[0]);
        assert_eq!(cb_net_usleep(&mut ctx), 0);

        ctx.write_bytes(BASE + 0x200, &[0xAA; 6]);
        args(&mut ctx, &[BASE + 0x200, 0]);
        assert_eq!(cb_get_mac_address(&mut ctx), 0);
        assert_eq!(ctx.read_bytes(BASE + 0x200, 6).unwrap(), vec![0; 6]);

        // Guest sockaddr conversion: length, then family.
        ctx.write_bytes(BASE + 0x300, &guest_sockaddr_in([127, 0, 0, 1], 80));
        assert_eq!(convert_guest_sockaddr(&ctx, BASE + 0x300, 1), Err(EINVAL_));
        assert_eq!(convert_guest_sockaddr(&ctx, BASE + 0x300, 7), Err(EINVAL_));
        assert_eq!(convert_guest_sockaddr(&ctx, BASE + 0x300, 8), Ok((Ipv4Addr::LOCALHOST, 80)));
        assert_eq!(convert_guest_sockaddr(&ctx, 0, 16), Err(EFAULT_));
        ctx.write_bytes(BASE + 0x301, &[28]);
        assert_eq!(convert_guest_sockaddr(&ctx, BASE + 0x300, 16), Err(EOPNOTSUPP_));
        ctx.write_bytes(BASE + 0x301, &[1]);
        assert_eq!(convert_guest_sockaddr(&ctx, BASE + 0x300, 16), Err(EAFNOSUPPORT_));

        assert_eq!(message_flags(MSG_PEEK | MSG_DONTWAIT), Ok(libc::MSG_PEEK | libc::MSG_DONTWAIT));
        assert_eq!(message_flags(0x1), Err(EOPNOTSUPP_));
        assert_eq!(map_errno(libc::EAGAIN), EWOULDBLOCK_);
        assert_eq!(map_errno(libc::EHOSTDOWN), 64);
        assert_eq!(map_errno(libc::ENOENT), EIO_);
    }

    /// Socket creation, argument order and address reporting.
    #[test]
    fn socket_validation_and_names() {
        let _ids = socket_ids();
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5D02_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        args(&mut ctx, &[0, 1, 2, 0]);
        assert_eq!(cb_socket(&mut ctx), net_error(EAFNOSUPPORT_) as i64);
        args(&mut ctx, &[0, 2, SOCK_DGRAM_P2P as u64, 17]);
        assert_eq!(cb_socket(&mut ctx), net_error(EPROTONOSUPPORT_) as i64);
        args(&mut ctx, &[0, 2, 2, 0]);
        let s = cb_socket(&mut ctx) as u64;
        assert!((SOCKET_FD_MIN as u64..SOCKET_FD_MAX as u64).contains(&s));

        // IPPROTO_TCP: EINVAL on a datagram socket; on a stream TCP_NODELAY
        // reads back as TF_NODELAY (4), other options ENOPROTOOPT.
        ctx.write_u32(BASE + 0x80, 1);
        ctx.write_u32(BASE + 0x88, 4);
        args(&mut ctx, &[s, IPPROTO_TCP as u64, TCP_NODELAY as u64, BASE + 0x80, 4]);
        assert_eq!(cb_setsockopt(&mut ctx), net_error(EINVAL_) as i64);
        args(&mut ctx, &[s, IPPROTO_TCP as u64, TCP_NODELAY as u64, BASE + 0x84, BASE + 0x88]);
        assert_eq!(cb_getsockopt(&mut ctx), net_error(EINVAL_) as i64);
        args(&mut ctx, &[0, 2, 1, 0]);
        let t = cb_socket(&mut ctx) as u64;
        // A stream socket created without a protocol reports TCP.
        ctx.write_u32(BASE + 0x88, 4);
        args(&mut ctx, &[t, SOL_SOCKET as u64, SO_PROTOCOL as u64, BASE + 0x84, BASE + 0x88]);
        assert_eq!(cb_getsockopt(&mut ctx), 0);
        assert_eq!(ctx.read_u32(BASE + 0x84), Some(libc::IPPROTO_TCP as u32));
        args(&mut ctx, &[t, IPPROTO_TCP as u64, TCP_NODELAY as u64, BASE + 0x80, 2]);
        assert_eq!(cb_setsockopt(&mut ctx), net_error(EINVAL_) as i64);
        // TCP_MAXSEG keeps the console's state: a socket that is not
        // connected refuses every segment with EINVAL and reads 536.
        args(&mut ctx, &[t, IPPROTO_TCP as u64, 2, BASE + 0x80, 4]);
        assert_eq!(cb_setsockopt(&mut ctx), net_error(EINVAL_) as i64);
        ctx.write_u32(BASE + 0x80, 536);
        args(&mut ctx, &[t, IPPROTO_TCP as u64, 2, BASE + 0x80, 4]);
        assert_eq!(cb_setsockopt(&mut ctx), net_error(EINVAL_) as i64);
        ctx.write_u32(BASE + 0x84, 4);
        args(&mut ctx, &[t, IPPROTO_TCP as u64, 2, BASE + 0x80, BASE + 0x84]);
        assert_eq!(cb_getsockopt(&mut ctx), 0);
        assert_eq!(ctx.read_u32(BASE + 0x80), Some(536));
        for (set, got) in [(1u32, 4u32), (0, 0)] {
            ctx.write_u32(BASE + 0x80, set);
            args(&mut ctx, &[t, IPPROTO_TCP as u64, TCP_NODELAY as u64, BASE + 0x80, 4]);
            assert_eq!(cb_setsockopt(&mut ctx), 0);
            ctx.write_u32(BASE + 0x88, 4);
            args(&mut ctx, &[t, IPPROTO_TCP as u64, TCP_NODELAY as u64, BASE + 0x84, BASE + 0x88]);
            assert_eq!(cb_getsockopt(&mut ctx), 0);
            assert_eq!(ctx.read_u32(BASE + 0x84), Some(got));
            // A shorter length gets the value cut to it.
            for capacity in [0u32, 1, 2, 3, 8] {
                ctx.write_bytes(BASE + 0x90, &[0xa5; 8]);
                ctx.write_u32(BASE + 0x88, capacity);
                args(&mut ctx, &[t, IPPROTO_TCP as u64, TCP_NODELAY as u64, BASE + 0x90, BASE + 0x88]);
                assert_eq!(cb_getsockopt(&mut ctx), 0, "capacity {capacity}");
                let n = capacity.min(4) as usize;
                assert_eq!(ctx.read_u32(BASE + 0x88), Some(n as u32));
                let bytes = ctx.read_bytes(BASE + 0x90, 8).unwrap();
                assert_eq!(&bytes[..n], &got.to_le_bytes()[..n]);
                assert!(bytes[n..].iter().all(|&b| b == 0xa5));
            }
        }
        args(&mut ctx, &[t]);
        assert_eq!(cb_socket_close(&mut ctx), 0);

        // NULL buffers fault before the descriptor is looked up.
        args(&mut ctx, &[9999, 0, 0, 0]);
        assert_eq!(cb_send(&mut ctx), net_error(EFAULT_) as i64);
        args(&mut ctx, &[s, BASE, 4, 0x1]);
        assert_eq!(cb_recv(&mut ctx), net_error(EOPNOTSUPP_) as i64);
        args(&mut ctx, &[9999, 0, 16]);
        assert_eq!(cb_bind(&mut ctx), net_error(EFAULT_) as i64);

        ctx.write_bytes(BASE, &guest_sockaddr_in([127, 0, 0, 1], 0));
        args(&mut ctx, &[s, BASE, 16]);
        assert_eq!(cb_bind(&mut ctx), 0);
        // A short address buffer receives the start of the address and its
        // length; nothing past it is written.
        ctx.write_bytes(BASE + 0x20, &[0xA5; 16]);
        ctx.write_u32(BASE + 0x40, 8);
        args(&mut ctx, &[s, BASE + 0x20, BASE + 0x40]);
        assert_eq!(cb_getsockname(&mut ctx), 0);
        assert_eq!(ctx.read_u32(BASE + 0x40), Some(8));
        assert_eq!(ctx.read_bytes(BASE + 0x24, 4).unwrap(), vec![127, 0, 0, 1]);
        assert_eq!(ctx.read_bytes(BASE + 0x28, 8).unwrap(), vec![0xA5; 8]);
        assert_eq!(ctx.read_bytes(BASE + 0x20, 2).unwrap(), vec![16, 2]);
        ctx.write_u32(BASE + 0x40, 0x7FFF_FFFF);
        assert_eq!(cb_getsockname(&mut ctx), 0);
        assert_eq!(ctx.read_u32(BASE + 0x40), Some(16));
        args(&mut ctx, &[s, 0, BASE + 0x40]);
        assert_eq!(cb_getsockname(&mut ctx), net_error(EINVAL_) as i64);
        // No sender address (a stream socket): the length becomes 0 and the
        // address is left alone.
        ctx.write_u32(BASE + 0x40, 16);
        ctx.write_bytes(BASE + 0x60, &[0xA5; 16]);
        assert_eq!(convert_host_sockaddr(&ctx, None, BASE + 0x60, BASE + 0x40), Ok(0));
        assert_eq!((ctx.read_u32(BASE + 0x40), ctx.read_bytes(BASE + 0x60, 16).unwrap()), (Some(0), vec![0xA5; 16]));
        args(&mut ctx, &[s, BASE, 4, 0, BASE + 0x20, 0]);
        assert_eq!(cb_recvfrom(&mut ctx), net_error(EINVAL_) as i64, "an address without a length");

        // Options: unknown names are ENOPROTOOPT, the send timeout is in us.
        // SO_PROTOCOL reports the creation protocol — UDP for this datagram
        // socket created without one — truncated to the caller's length.
        ctx.write_u32(BASE + 0x88, 4);
        args(&mut ctx, &[s, SOL_SOCKET as u64, SO_PROTOCOL as u64, BASE + 0x84, BASE + 0x88]);
        assert_eq!(cb_getsockopt(&mut ctx), 0);
        assert_eq!(ctx.read_u32(BASE + 0x84), Some(libc::IPPROTO_UDP as u32));
        ctx.write_bytes(BASE + 0x84, &[0xEE; 2]);
        ctx.write_u32(BASE + 0x88, 2);
        args(&mut ctx, &[s, SOL_SOCKET as u64, SO_PROTOCOL as u64, BASE + 0x84, BASE + 0x88]);
        assert_eq!(cb_getsockopt(&mut ctx), 0);
        assert_eq!((ctx.read_bytes(BASE + 0x84, 2).unwrap(), ctx.read_u32(BASE + 0x88)), ((libc::IPPROTO_UDP as u32).to_le_bytes()[..2].to_vec(), Some(2)));
        ctx.write_u32(BASE + 0x80, 2_000_000);
        args(&mut ctx, &[s, SOL_SOCKET as u64, SO_SNDTIMEO_USEC as u64, BASE + 0x80, 4]);
        assert_eq!(cb_setsockopt(&mut ctx), 0);
        ctx.write_u32(BASE + 0x88, 4);
        args(&mut ctx, &[s, SOL_SOCKET as u64, SO_SNDTIMEO_USEC as u64, BASE + 0x84, BASE + 0x88]);
        assert_eq!(cb_getsockopt(&mut ctx), 0);
        assert_eq!(ctx.read_u32(BASE + 0x84), Some(2_000_000));
        // SO_RCVTIMEO takes a struct timeval and reads back through both
        // options; 0x1106 takes microseconds.
        ctx.write_u64(BASE + 0xA0, 1);
        ctx.write_u64(BASE + 0xA8, 250_000);
        args(&mut ctx, &[s, SOL_SOCKET as u64, SO_RCVTIMEO as u64, BASE + 0xA0, 16]);
        assert_eq!(cb_setsockopt(&mut ctx), 0);
        ctx.write_u32(BASE + 0x88, 4);
        args(&mut ctx, &[s, SOL_SOCKET as u64, SO_RCVTIMEO_USEC as u64, BASE + 0x84, BASE + 0x88]);
        assert_eq!(cb_getsockopt(&mut ctx), 0);
        assert_eq!(ctx.read_u32(BASE + 0x84), Some(1_250_000));
        ctx.write_bytes(BASE + 0xB0, &[0xFF; 16]);
        ctx.write_u32(BASE + 0x88, 32);
        args(&mut ctx, &[s, SOL_SOCKET as u64, SO_RCVTIMEO as u64, BASE + 0xB0, BASE + 0x88]);
        assert_eq!(cb_getsockopt(&mut ctx), 0);
        assert_eq!((ctx.read_u64(BASE + 0xB0), ctx.read_u64(BASE + 0xB8), ctx.read_u32(BASE + 0x88)), (Some(1), Some(250_000), Some(16)));
        ctx.write_u64(BASE + 0xA0, 0);
        ctx.write_u64(BASE + 0xA8, 1_000_000);
        args(&mut ctx, &[s, SOL_SOCKET as u64, SO_RCVTIMEO as u64, BASE + 0xA0, 16]);
        assert_eq!(cb_setsockopt(&mut ctx), net_error(EDOM_) as i64);
        ctx.write_u64(BASE + 0xA0, u64::MAX);
        ctx.write_u64(BASE + 0xA8, 0);
        assert_eq!(cb_setsockopt(&mut ctx), net_error(EDOM_) as i64);
        args(&mut ctx, &[s, SOL_SOCKET as u64, SO_RCVTIMEO as u64, BASE + 0x80, 4]);
        assert_eq!(cb_setsockopt(&mut ctx), net_error(EINVAL_) as i64, "an int is not a timeval");
        ctx.write_u32(BASE + 0x80, 50_000);
        args(&mut ctx, &[s, SOL_SOCKET as u64, SO_RCVTIMEO_USEC as u64, BASE + 0x80, 4]);
        assert_eq!(cb_setsockopt(&mut ctx), 0);
        ctx.write_u32(BASE + 0x80, u32::MAX);
        assert_eq!(cb_setsockopt(&mut ctx), net_error(EINVAL_) as i64, "a negative timeout");
        assert_eq!(guest_timeval_usec(i64::MAX, 999_999), Ok(i32::MAX));
        ctx.write_u32(BASE + 0x80, 2_000_000);
        args(&mut ctx, &[s, SOL_SOCKET as u64, 0x7777, BASE + 0x80, 4]);
        assert_eq!(cb_setsockopt(&mut ctx), net_error(ENOPROTOOPT_) as i64);
        args(&mut ctx, &[s, SOL_SOCKET as u64, SO_REUSEADDR as u64, 0, 4]);
        assert_eq!(cb_setsockopt(&mut ctx), net_error(EFAULT_) as i64);

        // Epoll control argument order.
        ctx.write_bytes(BASE + 0x100, b"ep\0");
        args(&mut ctx, &[0, 0]);
        assert_eq!(cb_epoll_create(&mut ctx), net_error(EINVAL_) as i64);
        args(&mut ctx, &[BASE + 0x100, 0]);
        let ep = cb_epoll_create(&mut ctx) as u64;
        assert!(ep >= EPOLL_ID_BASE as u64);
        args(&mut ctx, &[ep, NET_EPOLL_CTL_DEL as u64, s, BASE + 0x180]);
        assert_eq!(cb_epoll_control(&mut ctx), net_error(EINVAL_) as i64);
        args(&mut ctx, &[ep, NET_EPOLL_CTL_DEL as u64, 9999, 0]);
        assert_eq!(cb_epoll_control(&mut ctx), NET_ERROR_BAD_FD as i64);
        args(&mut ctx, &[ep, NET_EPOLL_CTL_MOD as u64, s, BASE + 0x180]);
        assert_eq!(cb_epoll_control(&mut ctx), net_error(ENOENT_) as i64);
        args(&mut ctx, &[ep]);
        assert_eq!(cb_epoll_destroy(&mut ctx), 0);
        args(&mut ctx, &[s]);
        assert_eq!(cb_socket_close(&mut ctx), 0);
        assert_eq!(cb_socket_close(&mut ctx), NET_ERROR_BAD_FD as i64);
    }

    /// UDPP2P sockets: virtual port rules, shared transport, logical options.
    #[test]
    fn udp_p2p_sockets() {
        let _ids = socket_ids();
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5D03_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        args(&mut ctx, &[0, 2, SOCK_DGRAM_P2P as u64, 0]);
        let a = cb_socket(&mut ctx) as u64;
        let b = cb_socket(&mut ctx) as u64;
        let c = cb_socket(&mut ctx) as u64;
        let p2p_addr = |vport: u16| {
            let mut sa = guest_sockaddr_in([127, 0, 0, 1], P2P_UDP_PORT);
            sa[8..10].copy_from_slice(&vport.to_be_bytes());
            sa
        };
        ctx.write_bytes(BASE, &p2p_addr(0));
        args(&mut ctx, &[a, BASE, 16]);
        assert_eq!(cb_bind(&mut ctx), net_error(EOPNOTSUPP_) as i64);
        ctx.write_bytes(BASE, &p2p_addr(5353));
        assert_eq!(cb_bind(&mut ctx), net_error(EACCES_) as i64);
        ctx.write_bytes(BASE, &p2p_addr(1000));
        args(&mut ctx, &[a, BASE, 12]);
        assert_eq!(cb_bind(&mut ctx), net_error(EINVAL_) as i64);
        let mut wrong_port = p2p_addr(1000);
        wrong_port[2..4].copy_from_slice(&1234u16.to_be_bytes());
        ctx.write_bytes(BASE + 0x20, &wrong_port);
        args(&mut ctx, &[a, BASE + 0x20, 16]);
        assert_eq!(cb_bind(&mut ctx), net_error(EINVAL_) as i64);

        args(&mut ctx, &[a, BASE, 16]);
        let first = cb_bind(&mut ctx);
        if first != 0 {
            // UDP port 3658 is taken on this host; nothing more to check.
            assert_eq!(first, net_error(EADDRINUSE_) as i64);
            return;
        }
        assert_eq!(cb_bind(&mut ctx), net_error(EINVAL_) as i64);
        args(&mut ctx, &[b, BASE, 16]);
        assert_eq!(cb_bind(&mut ctx), net_error(EADDRINUSE_) as i64);
        ctx.write_bytes(BASE + 0x40, &p2p_addr(1001));
        args(&mut ctx, &[c, BASE + 0x40, 16]);
        assert_eq!(cb_bind(&mut ctx), 0);
        {
            let st = net();
            assert!(Arc::ptr_eq(&st.sockets[&(a as i32)].transport, &st.sockets[&(c as i32)].transport));
        }
        ctx.write_u32(BASE + 0x90, 16);
        args(&mut ctx, &[c, BASE + 0x80, BASE + 0x90]);
        assert_eq!(cb_getsockname(&mut ctx), 0);
        assert_eq!(ctx.read_bytes(BASE + 0x88, 2).unwrap(), 1001u16.to_be_bytes().to_vec());

        // Native-only operations are refused; options are per logical socket.
        args(&mut ctx, &[c, BASE + 0x100, 4, 0]);
        assert_eq!(cb_send(&mut ctx), net_error(EOPNOTSUPP_) as i64);
        ctx.write_u32(BASE + 0xA0, 1 << 20);
        args(&mut ctx, &[c, SOL_SOCKET as u64, SO_RCVBUF as u64, BASE + 0xA0, 4]);
        assert_eq!(cb_setsockopt(&mut ctx), 0);
        ctx.write_u32(BASE + 0xA8, 4);
        args(&mut ctx, &[c, SOL_SOCKET as u64, SO_RCVBUF as u64, BASE + 0xA4, BASE + 0xA8]);
        assert_eq!(cb_getsockopt(&mut ctx), 0);
        assert_eq!(ctx.read_u32(BASE + 0xA4), Some(512 * 1024));
        args(&mut ctx, &[a, SOL_SOCKET as u64, SO_RCVBUF as u64, BASE + 0xA4, BASE + 0xA8]);
        assert_eq!(cb_getsockopt(&mut ctx), 0);
        assert_eq!(ctx.read_u32(BASE + 0xA4), Some(40 * 1024));
        args(&mut ctx, &[c, IPPROTO_TCP as u64, TCP_NODELAY as u64, BASE + 0xA0, 4]);
        assert_eq!(cb_setsockopt(&mut ctx), net_error(EINVAL_) as i64);
        for s in [a, b, c] {
            args(&mut ctx, &[s]);
            assert_eq!(cb_socket_close(&mut ctx), 0);
        }
    }
}
