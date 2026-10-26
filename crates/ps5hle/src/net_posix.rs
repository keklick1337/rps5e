// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libkernel POSIX socket calls on stream servers and connected sockets:
//! listen/accept, send/recv, getpeername, shutdown and AF_UNIX socketpair.
//!
//! A listening kernel socket listens on its own host socket. Unless an
//! outbound redirect is configured, a listener bound to the wildcard
//! address (or not bound at all) is confined to loopback, so the guest
//! never exposes host ports to the network. Accepted connections become
//! new kernel socket descriptors.

#[cfg(windows)]
#[allow(unused_imports)]
use crate::host::c as libc;
use std::net::SocketAddr;

use super::*;

pub(super) const AF_UNIX: i32 = 1;

const SOCK_STREAM: i32 = 1;
const SOCK_DGRAM: i32 = 2;
const SOCK_SEQPACKET: i32 = 5;
const SOCK_CLOEXEC: i32 = 0x1000_0000;
const SOCK_NONBLOCK: i32 = 0x2000_0000;

const IPPROTO_TCP_: i32 = 6;
const IPPROTO_UDP_: i32 = 17;

const ENOMEM: i32 = 12;
const EFAULT: i32 = 14;
const EINVAL: i32 = 22;
const ENFILE: i32 = 23;
const EMFILE: i32 = 24;
const EPROTOTYPE: i32 = 41;
const EPROTONOSUPPORT: i32 = 43;
const EOPNOTSUPP: i32 = 45;
const EAFNOSUPPORT: i32 = 47;

/// The unnamed AF_UNIX address (sa_len 16, sa_family AF_UNIX) into a guest
/// (sockaddr*, socklen_t*) pair, cut to the given length.
pub(super) fn write_unnamed_address(ctx: &CpuContext, addr: u64, len_addr: u64) -> Result<(), i32> {
    if addr == 0 || len_addr == 0 {
        return Err(EFAULT);
    }
    let mut bytes = [0u8; 16];
    bytes[0] = 16;
    bytes[1] = AF_UNIX as u8;
    let n = ctx.read_u32(len_addr).ok_or(EFAULT)?.min(16);
    if (n != 0 && !ctx.write_bytes(addr, &bytes[..n as usize])) || !ctx.write_u32(len_addr, n) {
        return Err(EFAULT);
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn test_socketpair(ctx: &CpuContext, ty: i32, sv: u64) -> (i32, i32) {
    socketpair_core(ctx, AF_UNIX, ty, 0, sv).expect("socketpair");
    (ctx.read_u32(sv).unwrap() as i32, ctx.read_u32(sv + 4).unwrap() as i32)
}

/// socketpair(domain, type, protocol, int sv[2]): AF_UNIX stream, datagram
/// or sequenced-packet pairs with protocol 0; SOCK_NONBLOCK applies to both
/// ends, SOCK_CLOEXEC marks both close-on-exec. Nothing is written on
/// failure.
fn socketpair_core(ctx: &CpuContext, family: i32, ty: i32, protocol: i32, sv: u64) -> Result<(), i32> {
    let nonblocking = ty & SOCK_NONBLOCK != 0;
    let plain = ty & !(SOCK_CLOEXEC | SOCK_NONBLOCK);
    if family != AF_UNIX && family != ORBIS_AF_INET && family != ORBIS_AF_INET6 {
        return Err(EAFNOSUPPORT);
    }
    if plain == 0 {
        return Err(EPROTONOSUPPORT);
    }
    if family != AF_UNIX {
        if plain != SOCK_STREAM && plain != SOCK_DGRAM {
            return Err(EPROTOTYPE);
        }
        let native = if plain == SOCK_STREAM { IPPROTO_TCP_ } else { IPPROTO_UDP_ };
        return Err(if protocol == 0 || protocol == native { EOPNOTSUPP } else { EPROTONOSUPPORT });
    }
    if protocol != 0 {
        return Err(EPROTONOSUPPORT);
    }
    let native_type = match plain {
        SOCK_STREAM => libc::SOCK_STREAM,
        SOCK_DGRAM => libc::SOCK_DGRAM,
        SOCK_SEQPACKET => libc::SOCK_SEQPACKET,
        _ => return Err(EPROTOTYPE),
    };
    if sv == 0 || ctx.read_bytes(sv, 8).is_none() {
        return Err(EFAULT);
    }
    let mut native = [0i32; 2];
    let host_flags = libc::SOCK_CLOEXEC | if nonblocking { libc::SOCK_NONBLOCK } else { 0 };
    if unsafe { libc::socketpair(libc::AF_UNIX, native_type | host_flags, 0, native.as_mut_ptr()) } != 0 {
        return Err(match last_errno() {
            libc::EMFILE => EMFILE,
            libc::ENFILE => ENFILE,
            libc::ENOMEM => ENOMEM,
            e => map_errno(e),
        });
    }
    let mut fds = [0i32; 2];
    for (slot, end) in fds.iter_mut().zip(native) {
        let mut sock = KernelSocket::new(AF_UNIX, plain, HostSocket { fd: end });
        sock.connected = true;
        sock.nonblocking = nonblocking;
        *slot = install_socket(sock);
        if ty & SOCK_CLOEXEC != 0 {
            crate::filesystem::set_cloexec(*slot, true);
        }
    }
    // Each end knows its peer: getpeername reports ENOTCONN once the other
    // socket is closed (measured on hardware).
    {
        let mut socks = ksockets();
        let first = socks.get(&fds[0]).map(|s| s.host.clone());
        let second = socks.get(&fds[1]).map(|s| s.host.clone());
        if let (Some(first), Some(second)) = (first, second) {
            if let Some(a) = socks.get_mut(&fds[0]) {
                a.peer_host = Arc::downgrade(&second);
            }
            if let Some(b) = socks.get_mut(&fds[1]) {
                b.peer_host = Arc::downgrade(&first);
            }
        }
    }
    let mut out = [0u8; 8];
    out[..4].copy_from_slice(&fds[0].to_le_bytes());
    out[4..].copy_from_slice(&fds[1].to_le_bytes());
    if !ctx.write_bytes(sv, &out) {
        for fd in fds {
            kernel_socket_close(fd);
        }
        return Err(EFAULT);
    }
    net_log(format_args!("socketpair type={plain} nonblocking={nonblocking} -> {}, {}", fds[0], fds[1]));
    Ok(())
}

fn cb_kernel_socketpair(ctx: &mut CpuContext) -> HleResult {
    let r = socketpair_core(ctx, ctx.rdi() as i32, ctx.rsi() as i32, ctx.rdx() as i32, ctx.rcx());
    posix_socket_result(ctx, r.map(|()| 0))
}

/// listen(fd, backlog): stream sockets only (EOPNOTSUPP otherwise). An
/// unbound socket is first bound to a loopback port unless an outbound
/// redirect is configured.
fn kernel_listen(fd: i32, backlog: i32) -> Result<(), i32> {
    let s = socket_ref(fd)?;
    if s.ty != SOCK_STREAM {
        return Err(EOPNOTSUPP);
    }
    if s.family != AF_UNIX && !s.connected && !net_redirect_configured() && host_name(s.host.fd, false)?.port() == 0 {
        host_bind(s.host.fd, &SocketAddr::new(loopback_of(s.family), 0))?;
    }
    // SAFETY: plain call on an owned descriptor.
    if unsafe { libc::listen(s.host.fd, backlog) } < 0 {
        return Err(map_errno(last_errno()));
    }
    if let Some(k) = ksockets().get_mut(&fd) {
        k.listening = true;
    }
    net_log(format_args!("listen: fd={fd} on {:?}", host_name(s.host.fd, false)));
    Ok(())
}

pub(super) fn cb_kernel_listen(ctx: &mut CpuContext) -> HleResult {
    let r = kernel_listen(ctx.rdi() as i32, ctx.rsi() as i32);
    posix_socket_result(ctx, r.map(|()| 0))
}

/// accept(fd, addr*, addrlen*) -> new descriptor, blocking unless the
/// listener is non-blocking. The new socket inherits the listener's
/// non-blocking mode, SO_NOSIGPIPE and timeouts. The peer address is
/// written as far as `*addrlen` allows; if that fails the new descriptor is
/// closed again (EFAULT).
fn kernel_accept(ctx: &CpuContext, fd: i32, addr: u64, len_addr: u64) -> Result<i32, i32> {
    let s = socket_ref(fd)?;
    if s.ty != SOCK_STREAM || (addr != 0 && len_addr == 0) {
        return Err(EINVAL);
    }
    let mut ss: libc::sockaddr_storage = unsafe { std::mem::zeroed() };
    let mut len: libc::socklen_t = 0;
    let native = retry(|| unsafe {
        len = std::mem::size_of::<libc::sockaddr_storage>() as libc::socklen_t;
        libc::accept4(s.host.fd, &mut ss as *mut _ as *mut libc::sockaddr, &mut len, libc::SOCK_CLOEXEC) as isize
    })
    .map_err(map_errno)? as i32;
    let peer = sockaddr_to(&ss, len);
    let mut sock = KernelSocket::new(s.family, SOCK_STREAM, HostSocket { fd: native });
    sock.connected = true;
    {
        let socks = ksockets();
        if let Some(l) = socks.get(&fd) {
            sock.nonblocking = l.nonblocking;
            sock.no_sigpipe = l.no_sigpipe;
            sock.send_timeout = l.send_timeout;
            sock.receive_timeout = l.receive_timeout;
            // The console's own options cross accept: the kept flags, the
            // keep-alive values and the no-push TCP flag (measured on
            // hardware).
            sock.opts = SockOptState {
                kept_flags: l.opts.kept_flags,
                tcp_keep_idle: l.opts.tcp_keep_idle,
                tcp_keep_interval: l.opts.tcp_keep_interval,
                tcp_keep_count: l.opts.tcp_keep_count,
                tcp_flags: l.opts.tcp_flags & 0x1000,
                ..SockOptState::default()
            };
        }
    }
    set_host_nonblocking(native, sock.nonblocking)?;
    set_host_timeout(native, libc::SO_SNDTIMEO, sock.send_timeout)?;
    set_host_timeout(native, libc::SO_RCVTIMEO, sock.receive_timeout)?;
    let new_fd = install_socket(sock);
    net_log(format_args!("accept: fd={fd} -> {new_fd} from {peer:?}"));
    if addr != 0 {
        let peer = if s.family == AF_UNIX { None } else { peer };
        if let Err(e) = convert_host_sockaddr(ctx, peer, addr, len_addr) {
            kernel_socket_close(new_fd);
            return Err(e);
        }
    }
    Ok(new_fd)
}

pub(super) fn cb_kernel_accept(ctx: &mut CpuContext) -> HleResult {
    let r = kernel_accept(ctx, ctx.rdi() as i32, ctx.rsi(), ctx.rdx());
    posix_socket_result(ctx, r.map(|fd| fd as i64))
}

/// getpeername(fd, addr*, addrlen*)
pub(super) fn cb_kernel_getpeername(ctx: &mut CpuContext) -> HleResult {
    let r = socket_name(ctx, ctx.rdi() as i32, ctx.rsi(), ctx.rdx(), true);
    posix_socket_result(ctx, r.map(|()| 0))
}

/// recv(fd, buf, len, flags)
fn cb_kernel_recv(ctx: &mut CpuContext) -> HleResult {
    let r = receive_core(ctx, ctx.rdi() as i32, ctx.rsi(), ctx.rdx(), ctx.rcx() as i32, 0, 0);
    posix_socket_result(ctx, r)
}

/// shutdown(fd, how): 0 receive, 1 send, 2 both.
fn kernel_shutdown(fd: i32, how: i32) -> Result<(), i32> {
    let s = socket_ref(fd)?;
    let host_how = match how {
        0 => libc::SHUT_RD,
        1 => libc::SHUT_WR,
        2 => libc::SHUT_RDWR,
        _ => return Err(EINVAL),
    };
    // SAFETY: plain call on an owned descriptor.
    if unsafe { libc::shutdown(s.host.fd, host_how) } != 0 {
        return Err(map_errno(last_errno()));
    }
    if let Some(k) = ksockets().get_mut(&fd) {
        k.receive_shut |= how != 1;
        k.send_shut |= how != 0;
    }
    Ok(())
}

pub(super) fn cb_kernel_shutdown(ctx: &mut CpuContext) -> HleResult {
    let r = kernel_shutdown(ctx.rdi() as i32, ctx.rsi() as i32);
    posix_socket_result(ctx, r.map(|()| 0))
}

// ---------------------------------------------------------------------------
// libScePosixForWebKit: gethostbyname
// ---------------------------------------------------------------------------

/// h_errno values (reported through sceNetErrnoLoc).
const TRY_AGAIN: u32 = 2;
const NO_RECOVERY: u32 = 3;

/// Guest `struct hostent`: name, aliases, family (i32), length (i32),
/// address list; 32 bytes.
const HOSTENT_SIZE: u64 = 32;
/// Per-thread result block: the hostent, an empty alias list, a
/// one-address list, the address bytes and the name copy.
const HOSTENT_ALIASES: u64 = HOSTENT_SIZE;
const HOSTENT_ADDRESSES: u64 = HOSTENT_ALIASES + 8;
const HOSTENT_ADDRESS: u64 = HOSTENT_ADDRESSES + 16;
const HOSTENT_NAME: u64 = HOSTENT_ADDRESS + 8;
/// Longest host name kept (MAXHOSTNAMELEN plus the terminator).
const MAX_HOST_NAME: usize = 1025;

thread_local! {
    static HOSTENT_BLOCK: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

fn set_h_errno(ctx: &CpuContext, value: u32) {
    let slot = net_errno_address(ctx);
    if slot != 0 {
        ctx.write_u32(slot, value);
    }
}

/// gethostbyname(name) -> struct hostent* in per-thread storage (the next
/// call on the thread overwrites it), or NULL with h_errno set. A numeric
/// IPv4 name converts as FreeBSD's inet_aton reads it (one to four parts,
/// decimal, octal or hex, ending at white space) and clears the error; the
/// console is offline, so any other name fails with TRY_AGAIN.
fn cb_gethostbyname(ctx: &mut CpuContext) -> HleResult {
    let name_addr = ctx.rdi();
    let Some(name) = (name_addr != 0).then(|| ctx.read_cstr_bytes(name_addr, MAX_HOST_NAME)).flatten() else {
        set_h_errno(ctx, NO_RECOVERY);
        return 0;
    };
    let numeric = (name.len() < MAX_HOST_NAME).then(|| crate::libc_string::inet_aton_freebsd(&name)).flatten();
    let Some(address) = numeric else {
        log::info!("[net] gethostbyname('{}') -> no DNS (offline)", String::from_utf8_lossy(&name));
        set_h_errno(ctx, TRY_AGAIN);
        return 0;
    };
    let mut block = HOSTENT_BLOCK.with(|c| c.get());
    if block == 0 {
        let Some(p) = crate::pthread::alloc_guest_block(ctx, HOSTENT_NAME + MAX_HOST_NAME as u64) else {
            set_h_errno(ctx, NO_RECOVERY);
            return 0;
        };
        HOSTENT_BLOCK.with(|c| c.set(p));
        block = p;
    }
    let mut bytes = Vec::with_capacity(HOSTENT_NAME as usize + name.len() + 1);
    bytes.extend_from_slice(&(block + HOSTENT_NAME).to_le_bytes());
    bytes.extend_from_slice(&(block + HOSTENT_ALIASES).to_le_bytes());
    bytes.extend_from_slice(&(ORBIS_AF_INET as i32).to_le_bytes());
    bytes.extend_from_slice(&4i32.to_le_bytes());
    bytes.extend_from_slice(&(block + HOSTENT_ADDRESSES).to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&(block + HOSTENT_ADDRESS).to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&address.to_be_bytes());
    bytes.extend_from_slice(&[0; 4]);
    bytes.extend_from_slice(&name);
    bytes.push(0);
    if !ctx.write_bytes(block, &bytes) {
        set_h_errno(ctx, NO_RECOVERY);
        return 0;
    }
    set_h_errno(ctx, 0);
    block as HleResult
}

/// arc4random_buf(void* buf, size_t size): fill `size` bytes from the host
/// CSPRNG; nothing for size 0.
fn cb_arc4random_buf(ctx: &mut CpuContext) -> HleResult {
    let (buf, size) = (ctx.rdi(), ctx.rsi());
    let mut chunk = [0u8; 4096];
    let mut done = 0u64;
    while done < size {
        let n = (size - done).min(chunk.len() as u64) as usize;
        crate::misc_services::fill_random(&mut chunk[..n]);
        if !ctx.write_bytes(buf + done, &chunk[..n]) {
            log::error!("[libc] arc4random_buf: unwritable buffer {:#x}+{done:#x}", buf);
            break;
        }
        done += n as u64;
    }
    0
}

/// arc4random() -> a uniformly distributed 32-bit value from the same
/// source as arc4random_buf; there is no failure result.
fn cb_arc4random(_ctx: &mut CpuContext) -> HleResult {
    let mut b = [0u8; 4];
    crate::misc_services::fill_random(&mut b);
    u32::from_le_bytes(b) as HleResult
}

pub(super) const POSIX_WEBKIT_EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("5T4SHQqzoGo", "gethostbyname", cb_gethostbyname),
    ("n1KOc0lTUEs", "arc4random_buf", cb_arc4random_buf),
    ("UISQ-dQzw28", "arc4random", cb_arc4random),
];

pub(super) const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("pxnCmagrtao", "listen", cb_kernel_listen),
    ("3e+4Iv7IJ8U", "accept", cb_kernel_accept),
    ("TXFFFiNldU8", "getpeername", cb_kernel_getpeername),
    ("Ez8xjo9UF4E", "recv", cb_kernel_recv),
    ("TUuiYS2kE8s", "shutdown", cb_kernel_shutdown),
    ("MZb0GKT3mo8", "socketpair", cb_kernel_socketpair),
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CpuRegister::*;
    use std::io::{Read, Write};
    use std::net::TcpStream;

    #[test]
    fn nids_match_names() {
        for &(nid, name, _) in EXPORTS.iter().chain(POSIX_WEBKIT_EXPORTS) {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    /// Numeric names resolve as FreeBSD reads them, into the guest hostent
    /// (family 2, length 4, no aliases, one address) with h_errno cleared;
    /// other names fail offline.
    #[test]
    fn gethostbyname_numeric_ipv4() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5E67_0000_0000;
        vm.map(BASE, 0x2000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let h_errno = net_errno_address(&ctx);
        assert_ne!(h_errno, 0);
        let lookup = |ctx: &mut CpuContext, name: &[u8]| {
            ctx.write_bytes(BASE, name);
            ctx.write_bytes(BASE + name.len() as u64, &[0]);
            ctx.set(Rdi, BASE);
            ctx.write_u32(h_errno, 77);
            cb_gethostbyname(ctx) as u64
        };
        for (name, address) in [
            ("127.0.0.1", [127, 0, 0, 1]),
            ("127.1", [127, 0, 0, 1]),
            ("127.0.1", [127, 0, 0, 1]),
            ("0x7f.1", [127, 0, 0, 1]),
            ("0X7F.0x1", [127, 0, 0, 1]),
            ("0177.01", [127, 0, 0, 1]),
            ("0x7f000001", [127, 0, 0, 1]),
            ("2130706433", [127, 0, 0, 1]),
            ("127.1 trailing", [127, 0, 0, 1]),
            ("127.0.0.1\tignored", [127, 0, 0, 1]),
            ("4294967297", [0, 0, 0, 1]),
            ("18446744073709551617", [0, 0, 0, 1]),
            ("0x.1", [0, 0, 0, 1]),
            ("255.255.255.255", [255, 255, 255, 255]),
            ("0", [0, 0, 0, 0]),
            ("1.16777215", [1, 255, 255, 255]),
            ("1.2.65535", [1, 2, 255, 255]),
        ] {
            let entry = lookup(&mut ctx, name.as_bytes());
            assert_ne!(entry, 0, "{name:?}");
            let field = |off: u64| ctx.read_u64(entry + off).unwrap();
            assert_eq!(ctx.read_cstr_bytes(field(0), 64).unwrap(), name.as_bytes());
            assert_eq!(ctx.read_u64(field(8)), Some(0), "no aliases");
            assert_eq!(ctx.read_u32(entry + 16), Some(2));
            assert_eq!(ctx.read_u32(entry + 20), Some(4));
            let list = field(24);
            assert_eq!(ctx.read_bytes(ctx.read_u64(list).unwrap(), 4).unwrap(), address, "{name:?}");
            assert_eq!(ctx.read_u64(list + 8), Some(0));
            assert_eq!(ctx.read_u32(h_errno), Some(0));
        }
        // The per-thread result is reused.
        assert_eq!(lookup(&mut ctx, b"127.0.0.2"), lookup(&mut ctx, b"127.0.0.1"));
        // Names do not resolve offline; a null name is NO_RECOVERY.
        assert_eq!(lookup(&mut ctx, &[b'a'; 1024]), 0);
        assert!((1..=4).contains(&ctx.read_u32(h_errno).unwrap()));
        assert_eq!(lookup(&mut ctx, b"1.2.3.256"), 0);
        ctx.set(Rdi, 0);
        assert_eq!(cb_gethostbyname(&mut ctx), 0);
        assert_eq!(ctx.read_u32(h_errno), Some(3));
        vm.unmap(BASE, 0x2000).unwrap();
    }

    #[test]
    fn listen_accept_recv_shutdown_on_loopback() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x7A20_0000_0000;
        vm.map(BASE, 0x10000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);

        // socket(AF_INET, SOCK_STREAM, 0) + listen on an ephemeral port
        ctx.set(Rdi, ORBIS_AF_INET as u64);
        ctx.set(Rsi, SOCK_STREAM as u64);
        ctx.set(Rdx, 0);
        let fd = cb_kernel_socket(&mut ctx);
        assert!(fd > 2);
        ctx.set(Rdi, fd as u64);
        ctx.set(Rsi, 4);
        assert_eq!(cb_kernel_listen(&mut ctx), 0);
        let host = ksockets().get(&(fd as i32)).unwrap().host.fd;
        let local = host_name(host, false).unwrap();
        let (ip, port) = (local.ip(), local.port());
        assert!(ip.is_loopback());
        assert_ne!(port, 0);

        let client = std::thread::spawn(move || {
            let mut c = TcpStream::connect(SocketAddr::new(ip, port)).unwrap();
            c.write_all(b"ping").unwrap();
            let mut b = [0u8; 4];
            c.read_exact(&mut b).unwrap_or(());
        });

        ctx.set(Rdi, fd as u64);
        ctx.set(Rsi, BASE);
        ctx.write_u32(BASE + 0x20, 16);
        ctx.set(Rdx, BASE + 0x20);
        let conn = cb_kernel_accept(&mut ctx);
        // Any free number: another test may have closed a lower one.
        assert!(conn > 2 && conn != fd, "accept returned {conn}");
        assert_eq!(ctx.read_u32(BASE + 0x20), Some(16));
        assert_eq!(ctx.read_bytes(BASE + 4, 4).unwrap(), vec![127, 0, 0, 1]);

        ctx.set(Rdi, conn as u64);
        ctx.set(Rsi, BASE + 0x100);
        ctx.set(Rdx, 4);
        ctx.set(Rcx, MSG_WAITALL as u64);
        assert_eq!(cb_kernel_recv(&mut ctx), 4);
        assert_eq!(ctx.read_bytes(BASE + 0x100, 4).unwrap(), b"ping".to_vec());

        ctx.set(Rdi, conn as u64);
        ctx.set(Rsi, BASE + 0x200);
        ctx.write_u32(BASE + 0x220, 16);
        ctx.set(Rdx, BASE + 0x220);
        assert_eq!(cb_kernel_getpeername(&mut ctx), 0);
        assert_eq!(ctx.read_bytes(BASE + 0x204, 4).unwrap(), vec![127, 0, 0, 1]);

        ctx.set(Rdi, conn as u64);
        ctx.set(Rsi, 2);
        assert_eq!(cb_kernel_shutdown(&mut ctx), 0);
        client.join().unwrap();

        // Not-connected and unknown descriptors.
        ctx.set(Rdi, fd as u64);
        ctx.set(Rsi, BASE);
        ctx.set(Rdx, BASE + 0x20);
        assert_eq!(cb_kernel_getpeername(&mut ctx), -1);
        ctx.set(Rdi, 0x7FFF_0000);
        assert_eq!(cb_kernel_shutdown(&mut ctx), -1);
        ctx.set(Rdi, fd as u64);
        ctx.set(Rsi, 9);
        assert_eq!(cb_kernel_shutdown(&mut ctx), -1);

        assert!(kernel_socket_close(conn as i32));
        assert!(kernel_socket_close(fd as i32));
        assert!(!is_listening_socket(fd as i32));
    }

    /// Event-queue read and write events on kernel sockets: a listener
    /// becomes readable on a pending connection, a connected socket is
    /// writable with its send buffer size as data, readable with the bytes
    /// available, and reports EV_EOF once the peer is gone; a wait without a
    /// timeout wakes for data that arrives later; closing a socket drops its
    /// registrations.
    #[test]
    fn equeue_socket_read_and_write_events() {
        use crate::sync::equeue::{self, EV_EOF, FILTER_READ, FILTER_WRITE};
        let vm: &'static ps5vm::VirtualMemory = Box::leak(Box::new(ps5vm::VirtualMemory::new()));
        const BASE: u64 = 0x6A1B_0000_0000;
        vm.map(BASE, 0x10000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(vm);
        let eq_call = |ctx: &mut CpuContext, name: &str, args: &[u64]| {
            let cb = equeue::EXPORTS.iter().find(|e| e.1 == name).unwrap().3;
            for (r, v) in [Rdi, Rsi, Rdx, Rcx].iter().zip(args) {
                ctx.set(*r, *v);
            }
            cb(ctx)
        };
        let socket = |ctx: &mut CpuContext, protocol: u64| {
            ctx.set(Rdi, ORBIS_AF_INET as u64);
            ctx.set(Rsi, SOCK_STREAM as u64);
            ctx.set(Rdx, protocol);
            let fd = cb_kernel_socket(ctx);
            assert!(fd > 2);
            fd as u64
        };
        let listener = socket(&mut ctx, 6);
        ctx.set(Rdi, listener);
        ctx.set(Rsi, 4);
        assert_eq!(cb_kernel_listen(&mut ctx), 0);
        let local = host_name(ksockets().get(&(listener as i32)).unwrap().host.fd, false).unwrap();
        let (ip, port) = (local.ip(), local.port());
        let IpAddr::V4(v4) = ip else { panic!("listener on {ip}") };
        let address = BASE + 0x40;
        ctx.write_bytes(address, &guest_sockaddr_in(v4.octets(), port));
        let connect = |ctx: &mut CpuContext, fd: u64| {
            ctx.set(Rdi, fd);
            ctx.set(Rsi, address);
            ctx.set(Rdx, 16);
            assert_eq!(cb_kernel_connect(ctx), 0);
        };
        let accept = |ctx: &mut CpuContext, fd: u64| {
            ctx.set(Rdi, fd);
            ctx.set(Rsi, 0);
            ctx.set(Rdx, 0);
            let fd = cb_kernel_accept(ctx);
            assert!(fd > 2);
            fd as u64
        };
        let message = BASE + 0x80;
        ctx.write_bytes(message, b"ready\0");
        let send = |ctx: &mut CpuContext, fd: u64| {
            ctx.set(Rdi, fd);
            ctx.set(Rsi, message);
            ctx.set(Rdx, 6);
            ctx.set(Rcx, 0);
            assert_eq!(cb_posix_send(ctx), 6);
        };
        let recv = |ctx: &mut CpuContext, fd: u64| {
            ctx.set(Rdi, fd);
            ctx.set(Rsi, BASE + 0x100);
            ctx.set(Rdx, 6);
            ctx.set(Rcx, 0);
            assert_eq!(cb_kernel_recv(ctx), 6);
        };

        let eq = equeue::create_named("sockets");
        let marker = 0x5A5A;
        assert_eq!(eq_call(&mut ctx, "sceKernelAddReadEvent", &[eq, listener, 1, marker]), 0);
        assert!(equeue::wait(eq, 4, Some(20_000)).is_err(), "no connection yet");
        let client = socket(&mut ctx, 0);
        connect(&mut ctx, client);
        let evs = equeue::wait(eq, 4, Some(1_000_000)).unwrap();
        assert_eq!(evs.len(), 1);
        assert_eq!((evs[0].ident, evs[0].filter, evs[0].udata), (listener, FILTER_READ, marker));
        let accepted = accept(&mut ctx, listener);
        assert_eq!(eq_call(&mut ctx, "sceKernelDeleteReadEvent", &[eq, listener]), 0);
        assert_eq!(eq_call(&mut ctx, "sceKernelDeleteReadEvent", &[eq, listener]), 0x8002_0002u32 as i32 as i64);

        assert_eq!(eq_call(&mut ctx, "sceKernelAddWriteEvent", &[eq, client, 0, 0]), 0);
        let evs = equeue::wait(eq, 4, Some(1_000_000)).unwrap();
        assert_eq!(evs.len(), 1);
        assert!(evs[0].filter == FILTER_WRITE && evs[0].data > 0);
        assert_eq!(eq_call(&mut ctx, "sceKernelDeleteWriteEvent", &[eq, client]), 0);

        assert_eq!(eq_call(&mut ctx, "sceKernelAddReadEvent", &[eq, accepted, 1, 0]), 0);
        send(&mut ctx, client);
        let evs = equeue::wait(eq, 4, Some(1_000_000)).unwrap();
        assert_eq!(evs.len(), 1);
        assert_eq!((evs[0].ident, evs[0].data, evs[0].flags & EV_EOF), (accepted, 6, 0));
        recv(&mut ctx, accepted);
        assert!(equeue::wait(eq, 4, Some(20_000)).is_err(), "everything was read");
        assert!(kernel_socket_close(client as i32));
        let evs = equeue::wait(eq, 4, Some(1_000_000)).unwrap();
        assert_eq!(evs.len(), 1);
        assert!(evs[0].data == 0 && evs[0].flags & EV_EOF != 0, "{:?}", evs[0]);

        // A wait without a timeout wakes for data sent later, and for a
        // registration added while it sleeps.
        let blocking = equeue::create_named("blocking");
        let writer = socket(&mut ctx, 0);
        connect(&mut ctx, writer);
        let reader = accept(&mut ctx, listener);
        assert_eq!(eq_call(&mut ctx, "sceKernelAddReadEvent", &[blocking, reader, 1, 0]), 0);
        let sender = std::thread::spawn(move || {
            let mut ctx = CpuContext::new(vm);
            std::thread::sleep(std::time::Duration::from_millis(50));
            ctx.set(Rdi, writer);
            ctx.set(Rsi, message);
            ctx.set(Rdx, 6);
            ctx.set(Rcx, 0);
            assert_eq!(cb_posix_send(&mut ctx), 6);
        });
        let start = std::time::Instant::now();
        let evs = equeue::wait(blocking, 4, None).unwrap();
        sender.join().unwrap();
        assert!(start.elapsed() >= std::time::Duration::from_millis(40));
        assert_eq!(evs.len(), 1);
        assert_eq!((evs[0].ident, evs[0].data), (reader, 6));
        recv(&mut ctx, reader);
        let waiter = std::thread::spawn(move || equeue::wait(blocking, 4, None).unwrap());
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert_eq!(eq_call(&mut ctx, "sceKernelAddWriteEvent", &[blocking, writer, 0, 0]), 0);
        let evs = waiter.join().unwrap();
        assert_eq!(evs.len(), 1);
        assert_eq!((evs[0].ident, evs[0].filter), (writer, FILTER_WRITE));
        assert_eq!(eq_call(&mut ctx, "sceKernelDeleteWriteEvent", &[blocking, writer]), 0);
        assert!(kernel_socket_close(reader as i32));
        assert_eq!(eq_call(&mut ctx, "sceKernelDeleteReadEvent", &[blocking, reader]), 0x8002_0002u32 as i32 as i64);
        assert!(kernel_socket_close(writer as i32));
        assert!(equeue::delete(blocking));

        // A closed socket is no longer known.
        assert_eq!(eq_call(&mut ctx, "sceKernelAddReadEvent", &[eq, client, 1, 0]), 0x8002_0002u32 as i32 as i64);
        assert!(kernel_socket_close(accepted as i32));
        assert!(kernel_socket_close(listener as i32));
        assert!(equeue::delete(eq));
    }

    const MSG_NOSIGNAL_: u64 = 0x20000;
    const POLLIN_: i16 = 0x1;
    const POLLOUT_: i16 = 0x4;
    const POLLHUP_: i16 = 0x10;
    const EAGAIN_GUEST: i32 = 35;
    const EPIPE_GUEST: i32 = 32;

    struct Pair<'a> {
        ctx: CpuContext<'a>,
        base: u64,
    }

    impl Pair<'_> {
        fn create(&mut self, family: i32, ty: i32, protocol: i32) -> Result<(i32, i32), i32> {
            self.ctx.write_u64(self.base, u64::MAX);
            let r = socketpair_core(&self.ctx, family, ty, protocol, self.base);
            let v = self.ctx.read_u64(self.base).unwrap();
            match r {
                Ok(()) => Ok((v as u32 as i32, (v >> 32) as i32)),
                Err(e) => {
                    assert_eq!(v, u64::MAX, "a failed socketpair wrote descriptors");
                    Err(e)
                }
            }
        }

        fn send(&mut self, fd: i32, data: &[u8], flags: u64) -> Result<i64, i32> {
            self.ctx.write_bytes(self.base + 0x100, data);
            send_core(&self.ctx, fd, self.base + 0x100, data.len() as u64, flags as i32, 0, 0)
        }

        fn recv(&mut self, fd: i32, flags: u64) -> Result<Vec<u8>, i32> {
            let n = receive_core(&self.ctx, fd, self.base + 0x200, 32, flags as i32, 0, 0)?;
            Ok(self.ctx.read_bytes(self.base + 0x200, n as usize).unwrap())
        }

        /// poll() on (fd, events) pairs; returns the result and the revents.
        fn poll(&mut self, fds: &[(i32, i16)], timeout_ms: i32) -> (HleResult, Vec<i16>) {
            let at = self.base + 0x300;
            for (i, &(fd, events)) in fds.iter().enumerate() {
                let mut rec = [0u8; 8];
                rec[..4].copy_from_slice(&fd.to_le_bytes());
                rec[4..6].copy_from_slice(&events.to_le_bytes());
                self.ctx.write_bytes(at + i as u64 * 8, &rec);
            }
            self.ctx.set(Rdi, at);
            self.ctx.set(Rsi, fds.len() as u64);
            self.ctx.set(Rdx, timeout_ms as u32 as u64);
            let r = crate::filesystem::cb_poll(&mut self.ctx);
            let revents = (0..fds.len()).map(|i| self.ctx.read_u32(at + i as u64 * 8 + 4).unwrap() as i32 >> 16).map(|v| v as i16).collect();
            (r, revents)
        }

        fn readable(&mut self, fd: i32, accepted: i16) -> bool {
            let (r, revents) = self.poll(&[(fd, POLLIN_)], 2000);
            r == 1 && revents[0] & accepted != 0
        }

        fn exchange(&mut self, from: i32, to: i32, text: &[u8]) {
            assert_eq!(self.send(from, text, MSG_NOSIGNAL_), Ok(text.len() as i64));
            assert!(self.readable(to, POLLIN_));
            assert_eq!(self.recv(to, 0).unwrap(), text);
        }

        /// getsockname / getpeername report the unnamed AF_UNIX address,
        /// cut to the caller's length.
        fn unnamed(&mut self, fd: i32, query: fn(&mut CpuContext) -> HleResult) -> bool {
            let (name, len) = (self.base + 0x400, self.base + 0x480);
            let mut ok = true;
            for (capacity, end) in [(28u32, 16usize), (4, 4)] {
                self.ctx.write_bytes(name, &[0xAA; 28]);
                self.ctx.write_u32(len, capacity);
                self.ctx.set(Rdi, fd as u64);
                self.ctx.set(Rsi, name);
                self.ctx.set(Rdx, len);
                let b = self.ctx.read_bytes(name, 28).unwrap();
                ok &= query(&mut self.ctx) == 0;
                let b2 = self.ctx.read_bytes(name, 28).unwrap();
                ok &= b != b2 && self.ctx.read_u32(len) == Some(end as u32) && b2[0] == 16 && b2[1] == 1 && b2[end] == 0xAA;
            }
            ok
        }

        fn close(&self, fd: i32) -> bool {
            kernel_socket_close(fd)
        }
    }

    /// AF_UNIX pairs: data both ways, poll, peek, recvfrom without an
    /// address, O_NONBLOCK through fcntl and SOCK_NONBLOCK, shutdown and
    /// close seen by the peer, datagram and sequenced-packet boundaries,
    /// and the argument checks.
    #[test]
    fn socketpair_ends_are_connected_sockets() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x7A21_0000_0000;
        vm.map(BASE, 0x10000, ps5vm::MemProtect::RW).unwrap();
        let mut p = Pair { ctx: CpuContext::new(&vm), base: BASE };

        let (a, b) = p.create(AF_UNIX, SOCK_STREAM, 0).unwrap();
        assert_ne!(a, b);
        p.exchange(a, b, b"one way\0");
        p.exchange(b, a, b"other way\0");
        assert!(p.unnamed(a, crate::net::cb_kernel_getsockname));
        assert!(p.unnamed(b, cb_kernel_getpeername));

        let (r, revents) = p.poll(&[(b, POLLIN_), (a, POLLOUT_)], 0);
        assert_eq!(r, 1);
        assert_eq!(revents[0], 0);
        assert_ne!(revents[1] & POLLOUT_, 0);
        assert_eq!(p.send(a, b"ping\0", 0), Ok(5));
        let (r, revents) = p.poll(&[(b, POLLIN_)], 1000);
        assert!(r == 1 && revents[0] & POLLIN_ != 0);
        assert_eq!(p.recv(b, MSG_PEEK as u64).unwrap(), b"ping\0");
        // recvfrom on a stream end reports no address.
        p.ctx.write_u32(BASE + 0x580, 28);
        assert_eq!(receive_core(&p.ctx, b, BASE + 0x200, 8, 0, BASE + 0x500, BASE + 0x580), Ok(5));
        assert_eq!(p.ctx.read_u32(BASE + 0x580), Some(0));
        assert_eq!(p.ctx.read_bytes(BASE + 0x200, 5).unwrap(), b"ping\0");

        assert_eq!(crate::net::kernel_socket_fcntl(b, 3, 0), Some(Ok(2)));
        assert_eq!(crate::net::kernel_socket_fcntl(b, 4, 4), Some(Ok(0)));
        assert_eq!(crate::net::kernel_socket_fcntl(b, 3, 0), Some(Ok(6)));
        assert_eq!(p.recv(b, 0), Err(EAGAIN_GUEST));

        // Shutting down the write side reads as end of stream on the peer.
        p.ctx.set(Rdi, a as u64);
        p.ctx.set(Rsi, 1);
        assert_eq!(cb_kernel_shutdown(&mut p.ctx), 0);
        assert!(p.readable(b, POLLIN_ | POLLHUP_));
        assert_eq!(p.recv(b, 0).unwrap(), b"");
        p.exchange(b, a, b"still open\0");
        assert!(p.close(a));
        assert!(p.readable(b, POLLIN_ | POLLHUP_));
        assert_eq!(p.recv(b, 0).unwrap(), b"");
        assert_eq!(p.send(b, b"ping\0", MSG_NOSIGNAL_), Err(EPIPE_GUEST));
        // Without MSG_NOSIGNAL the host still raises no SIGPIPE.
        assert_eq!(p.send(b, b"ping\0", 0), Err(EPIPE_GUEST));
        assert!(p.close(b));
        assert!(!p.close(b));
        assert_eq!(p.send(b, b"x", 0), Err(EBADF_));

        let (a, b) = p.create(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0).unwrap();
        assert_eq!(crate::net::kernel_socket_fcntl(a, 3, 0), Some(Ok(6)));
        assert_eq!(crate::net::kernel_socket_fcntl(b, 3, 0), Some(Ok(6)));
        assert_eq!(p.recv(a, 0), Err(EAGAIN_GUEST));
        p.exchange(b, a, b"wake up\0");
        assert!(p.close(a) && p.close(b));

        let (a, b) = p.create(AF_UNIX, SOCK_DGRAM, 0).unwrap();
        assert_eq!(crate::net::kernel_socket_fcntl(a, 3, 0), Some(Ok(2)));
        assert_eq!(p.send(a, b"a", 0), Ok(1));
        assert_eq!(p.send(a, b"bb", 0), Ok(2));
        assert!(p.readable(b, POLLIN_));
        p.ctx.write_u32(BASE + 0x580, 28);
        assert_eq!(receive_core(&p.ctx, b, BASE + 0x200, 16, 0, BASE + 0x500, BASE + 0x580), Ok(1));
        assert_eq!(p.ctx.read_bytes(BASE + 0x200, 1).unwrap(), b"a");
        assert_eq!(p.ctx.read_u32(BASE + 0x580), Some(16));
        assert_eq!(p.ctx.read_bytes(BASE + 0x500, 2).unwrap(), vec![16, 1]);
        assert_eq!(p.recv(b, 0).unwrap(), b"bb");
        p.exchange(b, a, b"back\0");
        assert!(p.close(a) && p.close(b));

        let (a, b) = p.create(AF_UNIX, SOCK_SEQPACKET, 0).unwrap();
        assert_eq!(p.send(a, b"a", 0), Ok(1));
        assert_eq!(p.send(a, b"bb", 0), Ok(2));
        assert_eq!(p.recv(b, 0).unwrap().len(), 1);
        assert_eq!(p.recv(b, 0).unwrap().len(), 2);
        assert!(p.close(a) && p.close(b));

        assert_eq!(p.create(ORBIS_AF_INET, SOCK_STREAM, 0), Err(EOPNOTSUPP));
        assert_eq!(p.create(ORBIS_AF_INET6, SOCK_DGRAM, 0), Err(EOPNOTSUPP));
        assert_eq!(p.create(3, SOCK_STREAM, 0), Err(EAFNOSUPPORT));
        assert_eq!(p.create(AF_UNIX, 0, 0), Err(EPROTONOSUPPORT));
        assert_eq!(p.create(AF_UNIX, 3, 0), Err(EPROTOTYPE));
        assert_eq!(p.create(AF_UNIX, 3, 6), Err(EPROTONOSUPPORT));
        assert_eq!(p.create(AF_UNIX, SOCK_STREAM | 0x0100_0000, 0), Err(EPROTOTYPE));
        assert_eq!(p.create(AF_UNIX, SOCK_STREAM, 6), Err(EPROTONOSUPPORT));
        assert_eq!(socketpair_core(&p.ctx, AF_UNIX, SOCK_STREAM, 0, 0), Err(EFAULT));
    }

    /// arc4random_buf fills exactly the requested bytes with fresh random
    /// data.
    #[test]
    fn arc4random_buf_fills_exactly_size_bytes() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5F89_0000_0000;
        vm.map(BASE, 0x4000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let (first, second) = (BASE, BASE + 0x2000);
        let fill = |ctx: &mut CpuContext, at: u64, size: u64| {
            ctx.set(Rdi, at);
            ctx.set(Rsi, size);
            assert_eq!(cb_arc4random_buf(ctx), 0);
        };
        for size in [0u64, 1, 3, 5, 4098] {
            ctx.write_bytes(first, &[0xa5; 4099]);
            fill(&mut ctx, first, size);
            let bytes = ctx.read_bytes(first, 4099).unwrap();
            assert!(bytes[size as usize..].iter().all(|&b| b == 0xa5), "size {size} wrote past its end");
        }
        fill(&mut ctx, first, 4099);
        fill(&mut ctx, second, 4099);
        let (a, b) = (ctx.read_bytes(first, 4099).unwrap(), ctx.read_bytes(second, 4099).unwrap());
        assert_ne!(a, b);
        let mut counts = [0usize; 256];
        for v in a {
            counts[v as usize] += 1;
        }
        assert!(counts.iter().max().copied().unwrap() < 64);
        vm.unmap(BASE, 0x4000).unwrap();
    }

    /// arc4random returns 32-bit values (the upper half of RAX clear) that
    /// cover both halves of the range and do not repeat.
    #[test]
    fn arc4random_draws_32_bit_values() {
        let vm = ps5vm::VirtualMemory::new();
        let mut ctx = CpuContext::new(&vm);
        let values: Vec<i64> = (0..256).map(|_| cb_arc4random(&mut ctx)).collect();
        assert!(values.iter().all(|&v| (0..=u32::MAX as i64).contains(&v)));
        assert!(values.iter().any(|&v| v >= 1 << 31) && values.iter().any(|&v| v < 1 << 31));
        let distinct: std::collections::HashSet<_> = values.iter().collect();
        assert!(distinct.len() > 250);
    }
}
