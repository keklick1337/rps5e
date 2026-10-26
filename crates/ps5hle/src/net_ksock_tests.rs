// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libkernel socket behaviour as FreeBSD (and the console) shows it:
//! creation, connect/accept, transfers, options, descriptor control and
//! errno numbering. Loopback only.

#[cfg(windows)]
#[allow(unused_imports)]
use crate::host::c as libc;
use super::*;
use crate::context::CpuRegister;
use crate::filesystem::test_hooks as fs;
use ps5vm::{MemProtect, VirtualMemory};

const SOCKET: HleCallback = cb_kernel_socket;
const F_GETFD: u64 = 1;
const F_SETFD: u64 = 2;
const F_GETFL: u64 = 3;
const F_SETFL: u64 = 4;

/// A guest thread with scratch memory and a bound errno.
struct Guest {
    ctx: CpuContext<'static>,
    errno: u64,
    base: u64,
}

impl Guest {
    fn new(base: u64) -> Self {
        let vm: &'static VirtualMemory = Box::leak(Box::new(VirtualMemory::new()));
        vm.map(base, 0x10000, MemProtect::RW).unwrap();
        let ctx = CpuContext::new(vm);
        let errno = crate::kernel::bind_test_errno(&ctx);
        Self { ctx, errno, base }
    }

    /// Call `cb` with integer arguments: the result, or Err(errno) for -1.
    fn call(&mut self, cb: HleCallback, args: &[u64]) -> Result<i64, i32> {
        let regs = [CpuRegister::Rdi, CpuRegister::Rsi, CpuRegister::Rdx, CpuRegister::Rcx, CpuRegister::R8, CpuRegister::R9];
        for (r, v) in regs.iter().zip(args.iter().chain(std::iter::repeat(&0))) {
            self.ctx.set(*r, *v);
        }
        self.ctx.write_u32(self.errno, 0);
        let r = cb(&mut self.ctx);
        if r == -1 { Err(self.ctx.read_u32(self.errno).unwrap() as i32) } else { Ok(r) }
    }

    fn socket(&mut self, family: i32, ty: i32) -> i32 {
        self.call(SOCKET, &[family as u64, ty as u64, 0]).expect("socket") as i32
    }

    /// A guest sockaddr_in for 127.0.0.1:`port` at scratch offset `at`.
    fn loopback(&self, at: u64, port: u16) -> u64 {
        self.ctx.write_bytes(self.base + at, &guest_sockaddr_in([127, 0, 0, 1], port));
        self.base + at
    }

    fn name(&mut self, fd: i32) -> SocketAddr {
        let (addr, len) = (self.base + 0xF00, self.base + 0xF80);
        self.ctx.write_u32(len, 28);
        self.call(cb_kernel_getsockname, &[fd as u64, addr, len]).unwrap();
        let b = self.ctx.read_bytes(addr, 16).unwrap();
        SocketAddr::new(IpAddr::V4(Ipv4Addr::new(b[4], b[5], b[6], b[7])), u16::from_be_bytes([b[2], b[3]]))
    }

    /// A listening loopback stream socket and its address (at offset 0x40).
    fn listener(&mut self) -> (i32, u64) {
        let l = self.socket(2, 1);
        let a = self.loopback(0x40, 0);
        self.call(cb_kernel_bind, &[l as u64, a, 16]).unwrap();
        self.call(posix_ext::cb_kernel_listen, &[l as u64, 4]).unwrap();
        let port = self.name(l).port();
        (l, self.loopback(0x40, port))
    }

    fn set_int(&mut self, fd: i32, level: i32, option: i32, value: i32) -> Result<i64, i32> {
        self.ctx.write_u32(self.base + 0xE00, value as u32);
        self.call(cb_posix_setsockopt, &[fd as u64, level as u64, option as u64, self.base + 0xE00, 4])
    }

    fn get_int(&mut self, fd: i32, level: i32, option: i32) -> Result<i32, i32> {
        self.ctx.write_u32(self.base + 0xE10, 4);
        self.call(cb_posix_getsockopt, &[fd as u64, level as u64, option as u64, self.base + 0xE08, self.base + 0xE10])?;
        assert_eq!(self.ctx.read_u32(self.base + 0xE10), Some(4));
        Ok(self.ctx.read_u32(self.base + 0xE08).unwrap() as i32)
    }

    fn close(&self, fd: i32) {
        assert!(kernel_socket_close(fd), "close {fd}");
    }
}

fn on_thread(f: impl FnOnce() + Send + 'static) {
    std::thread::spawn(f).join().unwrap();
}

/// socket(): a type inet does not have is EPROTOTYPE, raw and
/// sequenced-packet ones and a wrong protocol EPROTONOSUPPORT, a family
/// other than inet EAFNOSUPPORT; SOCK_NONBLOCK and SOCK_CLOEXEC are
/// accepted in the type, and a closed socket's flag does not reach the
/// next socket on its number.
#[test]
fn socket_types_and_creation_flags() {
    on_thread(|| {
        let mut g = Guest::new(0x7E70_0000_0000);
        for family in [2u64, 28] {
            assert_eq!(g.call(SOCKET, &[family, 99, 0]), Err(EPROTOTYPE_));
            assert_eq!(g.call(SOCKET, &[family, 4, 0]), Err(EPROTOTYPE_));
            assert_eq!(g.call(SOCKET, &[family, 3, 0]), Err(EPROTONOSUPPORT_));
            assert_eq!(g.call(SOCKET, &[family, 0, 0]), Err(EPROTONOSUPPORT_));
            assert_eq!(g.call(SOCKET, &[family, 99, 6]), Err(EPROTONOSUPPORT_));
            assert_eq!(g.call(SOCKET, &[family, 1, 17]), Err(EPROTONOSUPPORT_));
            assert_eq!(g.call(SOCKET, &[family, 2, 6]), Err(EPROTONOSUPPORT_));
        }
        assert_eq!(g.call(SOCKET, &[99, 1, 0]), Err(EAFNOSUPPORT_));
        assert_eq!(g.call(SOCKET, &[1, 1, 0]), Err(EAFNOSUPPORT_));

        let plain = g.socket(2, 1);
        assert_eq!(fs::fcntl(plain, F_GETFL, 0), Ok(2));
        assert_eq!(fs::fcntl(plain, F_GETFD, 0), Ok(0));
        assert_eq!(fs::fcntl(plain, F_SETFD, 1), Ok(0));
        assert_eq!(fs::fcntl(plain, F_GETFD, 0), Ok(1));
        assert_eq!(fs::fcntl(plain, F_SETFD, 0), Ok(0));
        assert_eq!(fs::fcntl(plain, F_GETFD, 0), Ok(0));
        g.close(plain);

        let flagged = g.socket(2, 2 | SOCK_NONBLOCK_K | SOCK_CLOEXEC_K);
        assert_eq!(fs::fcntl(flagged, F_GETFL, 0), Ok(6));
        assert_eq!(fs::fcntl(flagged, F_GETFD, 0), Ok(1));
        // Non-blocking: an empty socket has nothing to read.
        assert_eq!(g.call(cb_posix_recvfrom, &[flagged as u64, g.base, 4, 0, 0, 0]), Err(EWOULDBLOCK_));
        g.close(flagged);
        let six = g.socket(28, 2);
        assert_eq!(g.get_int(six, IPPROTO_IPV6, IPV6_V6ONLY), Ok(1));
        g.close(six);
    });
}

/// connect() to port 0 is EADDRNOTAVAIL for a stream socket and an
/// unconnected datagram socket; a connected datagram socket keeps its own
/// error; a listener still answers EOPNOTSUPP first. __sys_socketex makes a
/// socket like socket().
/// Address lengths as FreeBSD's getsockaddr and the inet protocols check
/// them: above 255 ENAMETOOLONG, below 2 EINVAL (before a NULL address),
/// a short AF_INET / AF_INET6 address EINVAL.
#[test]
fn socket_address_lengths() {
    on_thread(|| {
        let mut g = Guest::new(0x7E7B_4000_0000);
        let fd = g.socket(2, 2);
        let a = g.loopback(0x100, 0);
        for (addr, len, errno) in [(a, 256u64, 63), (0, 256, 63), (0, 1, 22), (a, 0, 22), (0, 16, 14), (a, 15, 22)] {
            assert_eq!(g.call(cb_kernel_bind, &[fd as u64, addr, len]), Err(errno), "bind {addr:#x} {len}");
            assert_eq!(g.call(cb_kernel_connect, &[fd as u64, addr, len]), Err(errno), "connect {addr:#x} {len}");
        }
        g.ctx.write_bytes(g.base + 0x200, &[28, 28]);
        assert_eq!(g.call(cb_kernel_bind, &[fd as u64, g.base + 0x200, 27]), Err(22));
        g.ctx.write_bytes(g.base + 0x200, &[16, 7]);
        assert_eq!(g.call(cb_kernel_bind, &[fd as u64, g.base + 0x200, 16]), Err(47));
        assert_eq!(g.call(cb_kernel_bind, &[fd as u64, a, 16]), Ok(0));
        assert!(kernel_socket_close(fd));

        // Measured on a PS5 (firmware 13.60), datagram sockets: an AF_INET
        // socket takes exactly 16 bytes and an AF_INET6 one exactly 28
        // (EINVAL before the family); another family is EAFNOSUPPORT.
        let (udp, udp6, tcp6) = (g.socket(2, 2), g.socket(28, 2), g.socket(28, 1));
        let long = g.base + 0x400;
        let mut v4 = vec![0u8; 255];
        v4[..16].copy_from_slice(&guest_sockaddr_in([127, 0, 0, 1], 0));
        g.ctx.write_bytes(long, &v4);
        let data = g.base + 0x600;
        for len in [17u64, 28, 255] {
            assert_eq!(g.call(cb_kernel_bind, &[udp as u64, long, len]), Err(22), "bind {len}");
            assert_eq!(g.call(cb_kernel_connect, &[udp as u64, long, len]), Err(22), "connect {len}");
            assert_eq!(g.call(cb_posix_sendto, &[udp as u64, data, 1, 0, long, len]), Err(22), "sendto {len}");
        }
        let mut v6 = vec![0u8; 64];
        v6[..2].copy_from_slice(&[28, 28]);
        v6[23] = 1;
        let a6 = g.base + 0x700;
        g.ctx.write_bytes(a6, &v6);
        assert_eq!(g.call(cb_kernel_bind, &[tcp6 as u64, a6, 29]), Err(22));
        assert_eq!(g.call(cb_kernel_connect, &[tcp6 as u64, a6, 64]), Err(22));
        for family in [28u8, 1, 0] {
            g.ctx.write_bytes(long + 1, &[family]);
            assert_eq!(g.call(cb_kernel_bind, &[udp as u64, long, 16]), Err(47), "family {family}");
        }
        g.ctx.write_bytes(long + 1, &[28]);
        assert_eq!(g.call(cb_posix_sendto, &[udp as u64, data, 1, 0, long, 28]), Err(22));
        assert_eq!(g.call(cb_posix_sendto, &[udp as u64, data, 1, 0, long, 16]), Err(47));
        g.ctx.write_bytes(long + 1, &[2]);
        assert_eq!(g.call(cb_kernel_bind, &[udp6 as u64, long, 16]), Err(22));
        assert_eq!(g.call(cb_kernel_connect, &[udp6 as u64, long, 28]), Err(47));
        for fd in [udp, udp6, tcp6] {
            g.close(fd);
        }

        // A connected stream socket's sendto ignores an address once its
        // generic length rules pass.
        let (l, address) = g.listener();
        let c = g.socket(2, 1);
        g.call(cb_kernel_connect, &[c as u64, address, 16]).unwrap();
        let padded = g.base + 0x800;
        let mut p = g.ctx.read_bytes(address, 16).unwrap();
        p.push(0);
        g.ctx.write_bytes(padded, &p);
        assert_eq!(g.call(cb_posix_sendto, &[c as u64, data, 1, 0, padded, 17]), Ok(1));
        assert_eq!(g.call(cb_posix_sendto, &[c as u64, data, 1, 0, padded, 1]), Err(22));
        g.close(c);
        g.close(l);
    });
}

#[test]
fn connect_to_port_zero_and_socketex() {
    on_thread(|| {
        let mut g = Guest::new(0x7E7C_0000_0000);
        let zero = g.loopback(0x100, 0);
        let udp = g.socket(2, 2);
        assert_eq!(g.call(cb_kernel_connect, &[udp as u64, zero, 16]), Err(EADDRNOTAVAIL_));
        let tcp = g.socket(2, 1);
        assert_eq!(g.call(cb_kernel_connect, &[tcp as u64, zero, 16]), Err(EADDRNOTAVAIL_));
        g.close(tcp);
        // A datagram socket connected to a real port is not refused with 49.
        let peer = g.socket(2, 2);
        let a = g.loopback(0x140, 0);
        g.call(cb_kernel_bind, &[peer as u64, a, 16]).unwrap();
        let port = g.name(peer).port();
        let real = g.loopback(0x180, port);
        assert_eq!(g.call(cb_kernel_connect, &[udp as u64, real, 16]), Ok(0));
        assert_ne!(g.call(cb_kernel_connect, &[udp as u64, zero, 16]), Err(EADDRNOTAVAIL_));
        g.close(udp);
        g.close(peer);
        let (l, _) = g.listener();
        let zero = g.loopback(0x100, 0);
        assert_eq!(g.call(cb_kernel_connect, &[l as u64, zero, 16]), Err(EOPNOTSUPP_));
        g.close(l);
        let name = g.base + 0x200;
        g.ctx.write_bytes(name, b"named\0");
        let fd = g.call(cb_kernel_socketex, &[name, 2, 2, 0]).unwrap() as i32;
        assert!(is_kernel_socket(fd));
        g.close(fd);
    });
}

/// A listening socket's read readiness counts the queued connections.
#[test]
fn listener_read_event_counts_the_accept_queue() {
    on_thread(|| {
        let mut g = Guest::new(0x7E7D_0000_0000);
        let (l, address) = g.listener();
        let (a, b) = (g.socket(2, 1), g.socket(2, 1));
        assert_eq!(g.call(cb_kernel_connect, &[a as u64, address, 16]), Ok(0));
        assert_eq!(g.call(cb_kernel_connect, &[b as u64, address, 16]), Ok(0));
        let mut queued = 0;
        for _ in 0..200 {
            queued = socket_readiness(l, false).map_or(0, |r| r.available);
            if queued == 2 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(queued, 2);
        for fd in [a, b, l] {
            g.close(fd);
        }
    });
}

/// connect() on a listening socket checks the address like any connect
/// first (a null or one-byte address fails as on a plain socket), then
/// fails with EOPNOTSUPP, also for an address of the other family; the
/// listener keeps working.
#[test]
fn connect_on_a_listener_checks_the_address_first() {
    on_thread(|| {
        let mut g = Guest::new(0x7E71_0000_0000);
        let (l, address) = g.listener();
        assert_eq!(g.call(cb_kernel_connect, &[l as u64, address, 16]), Err(EOPNOTSUPP_));
        let plain = g.socket(2, 1);
        let null = g.call(cb_kernel_connect, &[plain as u64, 0, 16]).unwrap_err();
        assert_eq!(g.call(cb_kernel_connect, &[l as u64, 0, 16]), Err(null));
        let short = g.call(cb_kernel_connect, &[plain as u64, address, 1]).unwrap_err();
        assert_eq!(g.call(cb_kernel_connect, &[l as u64, address, 1]), Err(short));
        assert!(null != EOPNOTSUPP_ && short != EOPNOTSUPP_);
        g.close(plain);
        let six = g.base + 0x80;
        g.ctx.write_bytes(six, &[28, 28]);
        assert_eq!(g.call(cb_kernel_connect, &[l as u64, six, 28]), Err(EOPNOTSUPP_));

        let client = g.socket(2, 1);
        assert_eq!(g.call(cb_kernel_connect, &[client as u64, address, 16]), Ok(0));
        let accepted = g.call(posix_ext::cb_kernel_accept, &[l as u64, 0, 0]).unwrap() as i32;
        for fd in [accepted, client, l] {
            g.close(fd);
        }
    });
}

/// connect() to AF_UNSPEC dissolves a UDP association (and then fails
/// with EAFNOSUPPORT); the local port stays. A connected datagram socket
/// refuses sendto with an address (EISCONN); an unconnected one refuses
/// send (EDESTADDRREQ).
#[test]
fn connect_to_unspec_dissolves_a_datagram_association() {
    on_thread(|| {
        let mut g = Guest::new(0x7E72_0000_0000);
        let (a, b) = (g.socket(2, 2), g.socket(2, 2));
        for fd in [a, b] {
            let any = g.loopback(0x40, 0);
            g.call(cb_kernel_bind, &[fd as u64, any, 16]).unwrap();
        }
        let (port_a, port_b) = (g.name(a).port(), g.name(b).port());
        let to_b = g.loopback(0x60, port_b);
        let data = g.base + 0x100;
        g.ctx.write_bytes(data, b"x");
        assert_eq!(g.call(cb_posix_send, &[a as u64, data, 1, 0]), Err(EDESTADDRREQ_));
        assert_eq!(g.call(cb_kernel_connect, &[a as u64, to_b, 16]), Ok(0));
        assert_eq!(g.call(cb_posix_send, &[a as u64, data, 1, 0]), Ok(1));
        assert_eq!(g.call(cb_posix_sendto, &[a as u64, data, 1, 0, to_b, 16]), Err(EISCONN_));
        let unspec = g.base + 0x80;
        g.ctx.write_bytes(unspec, &[16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(g.call(cb_kernel_connect, &[a as u64, unspec, 16]), Err(EAFNOSUPPORT_));
        assert_eq!(g.call(cb_posix_send, &[a as u64, data, 1, 0]), Err(EDESTADDRREQ_));
        assert_eq!(g.name(a).port(), port_a);
        assert_eq!(g.call(cb_posix_sendto, &[a as u64, data, 1, 0, to_b, 16]), Ok(1));
        g.close(a);
        g.close(b);
    });
}

/// accept(): the new socket inherits the listener's non-blocking mode,
/// SO_NOSIGPIPE and timeouts; a blocking listener gives blocking sockets.
/// The peer address is cut to the caller's length, which then holds the
/// length copied.
#[test]
fn accepted_sockets_inherit_the_listener_mode() {
    on_thread(|| {
        let mut g = Guest::new(0x7E73_0000_0000);
        let (l, address) = g.listener();
        let accept = posix_ext::cb_kernel_accept;
        let c1 = g.socket(2, 1);
        g.call(cb_kernel_connect, &[c1 as u64, address, 16]).unwrap();
        let (peer, len) = (g.base + 0x200, g.base + 0x280);
        g.ctx.write_bytes(peer, &[0xAA; 16]);
        g.ctx.write_u32(len, 8);
        let s1 = g.call(accept, &[l as u64, peer, len]).unwrap() as i32;
        assert_eq!(fs::fcntl(s1, F_GETFL, 0), Ok(2));
        assert_eq!(g.ctx.read_u32(len), Some(8));
        assert_eq!(g.ctx.read_bytes(peer, 9).unwrap(), vec![16, 2, (g.name(c1).port() >> 8) as u8, g.name(c1).port() as u8, 127, 0, 0, 1, 0xAA]);

        assert_eq!(g.set_int(l, SOL_SOCKET, SO_NBIO, 1), Ok(0));
        assert_eq!(fs::fcntl(l, F_GETFL, 0), Ok(6));
        assert_eq!(g.call(accept, &[l as u64, 0, 0]), Err(EWOULDBLOCK_));
        assert_eq!(g.set_int(l, SOL_SOCKET, SO_NOSIGPIPE, 1), Ok(0));
        g.ctx.write_bytes(g.base + 0x300, &[3u64.to_le_bytes(), 0u64.to_le_bytes()].concat());
        g.call(cb_posix_setsockopt, &[l as u64, SOL_SOCKET as u64, SO_RCVTIMEO as u64, g.base + 0x300, 16]).unwrap();
        let c2 = g.socket(2, 1);
        g.call(cb_kernel_connect, &[c2 as u64, address, 16]).unwrap();
        let s2 = loop {
            match g.call(accept, &[l as u64, 0, 0]) {
                Ok(fd) => break fd as i32,
                Err(e) => assert_eq!(e, EWOULDBLOCK_),
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        };
        assert_eq!(fs::fcntl(s2, F_GETFL, 0), Ok(6));
        assert_eq!(g.get_int(s2, SOL_SOCKET, SO_NOSIGPIPE), Ok(SO_NOSIGPIPE));
        assert_eq!(g.get_int(s1, SOL_SOCKET, SO_NOSIGPIPE), Ok(0));
        g.ctx.write_u32(g.base + 0x330, 16);
        g.call(cb_posix_getsockopt, &[s2 as u64, SOL_SOCKET as u64, SO_RCVTIMEO as u64, g.base + 0x310, g.base + 0x330]).unwrap();
        assert_eq!(g.ctx.read_u64(g.base + 0x310), Some(3));
        assert_eq!(g.call(cb_posix_recvfrom, &[s2 as u64, g.base, 4, 0, 0, 0]), Err(EWOULDBLOCK_));
        for fd in [s1, s2, c1, c2, l] {
            g.close(fd);
        }
    });
}

/// TCP_MAXSEG as on a PS5 (firmware 13.60): a new socket reads 536 (1220
/// for AF_INET6) and refuses every segment; a connected one takes one up
/// to the current segment and raises one below 216 to 216.
#[test]
fn tcp_maximum_segment_needs_a_connection() {
    on_thread(|| {
        let mut g = Guest::new(0x7E7E_0000_0000);
        let fresh = g.socket(2, 1);
        for segment in [176, 300, 536] {
            assert_eq!(g.set_int(fresh, IPPROTO_TCP, TCP_MAXSEG, segment), Err(EINVAL_), "{segment}");
        }
        assert_eq!(g.get_int(fresh, IPPROTO_TCP, TCP_MAXSEG), Ok(536));
        let fresh6 = g.socket(28, 1);
        assert_eq!(g.get_int(fresh6, IPPROTO_TCP, TCP_MAXSEG), Ok(1220));
        let (l, address) = g.listener();
        let client = g.socket(2, 1);
        g.call(cb_kernel_connect, &[client as u64, address, 16]).unwrap();
        let server = g.call(posix_ext::cb_kernel_accept, &[l as u64, 0, 0]).unwrap() as i32;
        assert_eq!(g.get_int(server, IPPROTO_TCP, TCP_MAXSEG), Ok(536));
        assert_eq!(g.set_int(client, IPPROTO_TCP, TCP_MAXSEG, 176), Ok(0));
        assert_eq!(g.get_int(client, IPPROTO_TCP, TCP_MAXSEG), Ok(216));
        assert_eq!(g.set_int(client, IPPROTO_TCP, TCP_MAXSEG, 200), Ok(0));
        assert_eq!(g.get_int(client, IPPROTO_TCP, TCP_MAXSEG), Ok(216));
        assert_eq!(g.set_int(client, IPPROTO_TCP, TCP_MAXSEG, 400), Err(EINVAL_));
        assert_eq!(g.set_int(client, IPPROTO_TCP, TCP_MAXSEG, 175), Err(EINVAL_));
        for fd in [fresh, fresh6, client, server, l] {
            g.close(fd);
        }
    });
}

/// Transfers: MSG_DONTWAIT on an empty blocking socket is EWOULDBLOCK;
/// recvfrom on a stream socket reports no sender (length 0), on a datagram
/// socket the sender cut to the caller's length; getsockname and
/// getpeername report the length copied; recv on a never-connected stream
/// socket is ENOTCONN and so is send.
#[test]
fn transfers_flags_and_address_lengths() {
    on_thread(|| {
        let mut g = Guest::new(0x7E74_0000_0000);
        let (l, address) = g.listener();
        let c = g.socket(2, 1);
        g.call(cb_kernel_connect, &[c as u64, address, 16]).unwrap();
        let s = g.call(posix_ext::cb_kernel_accept, &[l as u64, 0, 0]).unwrap() as i32;
        let (buf, from, len) = (g.base + 0x100, g.base + 0x200, g.base + 0x280);
        assert_eq!(g.call(cb_posix_recvfrom, &[s as u64, buf, 8, MSG_DONTWAIT as u64, 0, 0]), Err(EWOULDBLOCK_));
        g.ctx.write_bytes(buf, b"abc");
        assert_eq!(g.call(cb_posix_send, &[c as u64, buf, 3, MSG_DONTWAIT as u64]), Ok(3));
        g.ctx.write_u32(len, 16);
        assert_eq!(g.call(cb_posix_recvfrom, &[s as u64, buf + 0x10, 8, 0, from, len]), Ok(3));
        assert_eq!(g.ctx.read_u32(len), Some(0));
        g.ctx.write_u32(len, 4);
        assert_eq!(g.call(posix_ext::cb_kernel_getpeername, &[s as u64, from, len]), Ok(0));
        assert_eq!(g.ctx.read_u32(len), Some(4));
        assert_eq!(g.ctx.read_bytes(from, 2).unwrap(), vec![16, 2]);
        g.ctx.write_u32(len, 64);
        assert_eq!(g.call(cb_kernel_getsockname, &[s as u64, from, len]), Ok(0));
        assert_eq!(g.ctx.read_u32(len), Some(16));

        let idle = g.socket(2, 1);
        assert_eq!(g.call(cb_posix_recvfrom, &[idle as u64, buf, 8, 0, 0, 0]), Err(ENOTCONN_));
        assert_eq!(g.call(cb_posix_send, &[idle as u64, buf, 1, 0]), Err(ENOTCONN_));
        assert_eq!(g.call(cb_posix_send, &[c as u64, buf, 1, 0x8000]), Err(EOPNOTSUPP_));

        let (u1, u2) = (g.socket(2, 2), g.socket(2, 2));
        for fd in [u1, u2] {
            let any = g.loopback(0x40, 0);
            g.call(cb_kernel_bind, &[fd as u64, any, 16]).unwrap();
        }
        let port = g.name(u2).port();
        let to = g.loopback(0x60, port);
        assert_eq!(g.call(cb_posix_sendto, &[u1 as u64, buf, 3, 0, to, 16]), Ok(3));
        g.ctx.write_u32(len, 6);
        g.ctx.write_bytes(from, &[0xAA; 16]);
        assert_eq!(g.call(cb_posix_recvfrom, &[u2 as u64, buf + 0x20, 8, 0, from, len]), Ok(3));
        assert_eq!(g.ctx.read_u32(len), Some(6));
        assert_eq!(g.ctx.read_bytes(from, 7).unwrap()[..2], [16, 2]);
        assert_eq!(g.ctx.read_bytes(from + 6, 1).unwrap(), vec![0xAA]);
        for fd in [u1, u2, idle, s, c, l] {
            g.close(fd);
        }
    });
}

/// Socket options as FreeBSD reports them: SO_TYPE, SO_ACCEPTCONN and the
/// boolean options as their own bit, SO_LINGER {bit, seconds} with EDOM
/// past 65535, values shorter than an int EINVAL and longer ones read from
/// their first int, short output buffers cut, unknown options and levels
/// ENOPROTOOPT, TCP_NODELAY on inet streams only (reported as 4),
/// SO_NBIO, SO_ERROR (cleared once read), SO_SNDBUF below 1 EINVAL, and a
/// read low-water mark above 1 refused on non-stream sockets.
#[test]
fn socket_options_follow_freebsd() {
    on_thread(|| {
        let mut g = Guest::new(0x7E75_0000_0000);
        let stream = g.socket(2, 1);
        let datagram = g.socket(2, 2);
        assert_eq!(g.get_int(stream, SOL_SOCKET, SO_TYPE), Ok(1));
        assert_eq!(g.get_int(datagram, SOL_SOCKET, SO_TYPE), Ok(2));
        assert_eq!(g.get_int(stream, SOL_SOCKET, SO_ACCEPTCONN), Ok(0));
        assert_eq!(g.get_int(stream, SOL_SOCKET, SO_KEEPALIVE), Ok(0));
        assert_eq!(g.set_int(stream, SOL_SOCKET, SO_KEEPALIVE, 1), Ok(0));
        assert_eq!(g.get_int(stream, SOL_SOCKET, SO_KEEPALIVE), Ok(SO_KEEPALIVE));
        assert_eq!(g.set_int(datagram, SOL_SOCKET, SO_BROADCAST, 7), Ok(0));
        assert_eq!(g.get_int(datagram, SOL_SOCKET, SO_BROADCAST), Ok(SO_BROADCAST));
        assert_eq!(g.set_int(stream, SOL_SOCKET, SO_NOSIGPIPE, 3), Ok(0));
        assert_eq!(g.get_int(stream, SOL_SOCKET, SO_NOSIGPIPE), Ok(SO_NOSIGPIPE));
        assert_eq!(g.set_int(stream, SOL_SOCKET, 0x7FFF, 1), Err(ENOPROTOOPT_));
        assert_eq!(g.get_int(stream, SOL_SOCKET, 0x7FFF), Err(ENOPROTOOPT_));
        assert_eq!(g.set_int(stream, 0, 1, 1), Err(ENOPROTOOPT_));

        // Lengths.
        let v = g.base + 0x100;
        g.ctx.write_u64(v, 1);
        assert_eq!(g.call(cb_posix_setsockopt, &[stream as u64, SOL_SOCKET as u64, SO_REUSEADDR as u64, v, 2]), Err(EINVAL_));
        assert_eq!(g.call(cb_posix_setsockopt, &[stream as u64, SOL_SOCKET as u64, SO_REUSEADDR as u64, v, 8]), Ok(0));
        g.ctx.write_bytes(v, &[0xAA; 4]);
        g.ctx.write_u32(v + 0x10, 2);
        assert_eq!(g.call(cb_posix_getsockopt, &[stream as u64, SOL_SOCKET as u64, SO_REUSEADDR as u64, v, v + 0x10]), Ok(0));
        assert_eq!(g.ctx.read_u32(v + 0x10), Some(2));
        assert_eq!(g.ctx.read_bytes(v, 3).unwrap(), vec![SO_REUSEADDR as u8, 0, 0xAA]);
        assert_eq!(g.set_int(stream, SOL_SOCKET, SO_SNDBUF, 0), Err(EINVAL_));
        assert_eq!(g.set_int(stream, SOL_SOCKET, SO_SNDBUF, 65536), Ok(0));
        assert_eq!(g.get_int(stream, SOL_SOCKET, SO_SNDBUF), Ok(65536));

        // SO_LINGER.
        let linger = g.base + 0x200;
        g.ctx.write_bytes(linger, &[1i32.to_le_bytes(), 5i32.to_le_bytes()].concat());
        assert_eq!(g.call(cb_posix_setsockopt, &[stream as u64, SOL_SOCKET as u64, SO_LINGER as u64, linger, 8]), Ok(0));
        g.ctx.write_u32(linger + 0x10, 8);
        assert_eq!(g.call(cb_posix_getsockopt, &[stream as u64, SOL_SOCKET as u64, SO_LINGER as u64, linger + 0x20, linger + 0x10]), Ok(0));
        assert_eq!(g.ctx.read_u32(linger + 0x20), Some(SO_LINGER as u32));
        assert_eq!(g.ctx.read_u32(linger + 0x24), Some(5));
        g.ctx.write_bytes(linger, &[1i32.to_le_bytes(), 65536i32.to_le_bytes()].concat());
        assert_eq!(g.call(cb_posix_setsockopt, &[stream as u64, SOL_SOCKET as u64, SO_LINGER as u64, linger, 8]), Err(EDOM_));
        assert_eq!(g.call(cb_posix_setsockopt, &[stream as u64, SOL_SOCKET as u64, SO_LINGER as u64, linger, 4]), Err(EINVAL_));

        // TCP level.
        assert_eq!(g.get_int(stream, IPPROTO_TCP, TCP_NODELAY), Ok(0));
        assert_eq!(g.set_int(stream, IPPROTO_TCP, TCP_NODELAY, 7), Ok(0));
        assert_eq!(g.get_int(stream, IPPROTO_TCP, TCP_NODELAY), Ok(TF_NODELAY));
        // TCP_MAXSEG keeps the console's state: 1 is below the smallest
        // allowed segment (216-40), refused with EINVAL.
        assert_eq!(g.set_int(stream, IPPROTO_TCP, 2, 1), Err(EINVAL_));
        assert_eq!(g.set_int(datagram, IPPROTO_TCP, TCP_NODELAY, 1), Err(EINVAL_));
        assert_eq!(g.get_int(datagram, IPPROTO_TCP, TCP_NODELAY), Err(EINVAL_));

        // SO_NBIO and the status flags agree.
        assert_eq!(g.set_int(datagram, SOL_SOCKET, SO_NBIO, 5), Ok(0));
        assert_eq!(g.get_int(datagram, SOL_SOCKET, SO_NBIO), Ok(1));
        assert_eq!(fs::fcntl(datagram, F_GETFL, 0), Ok(6));
        assert_eq!(fs::fcntl(datagram, F_SETFL, 2), Ok(0));
        assert_eq!(g.get_int(datagram, SOL_SOCKET, SO_NBIO), Ok(0));
        assert_eq!(fs::fcntl(datagram, F_SETFL, 2 | 0x8), Err(EOPNOTSUPP_));

        // Low-water marks.
        assert_eq!(g.set_int(datagram, SOL_SOCKET, SO_RCVLOWAT, 4), Err(EINVAL_));
        assert_eq!(g.set_int(datagram, SOL_SOCKET, SO_RCVLOWAT, 1), Ok(0));
        assert_eq!(g.set_int(stream, SOL_SOCKET, SO_RCVLOWAT, 0), Err(EINVAL_));
        assert_eq!(g.set_int(stream, SOL_SOCKET, SO_RCVLOWAT, 4), Ok(0));
        assert_eq!(g.get_int(stream, SOL_SOCKET, SO_RCVLOWAT), Ok(4));

        // SO_ERROR after a refused non-blocking connect, cleared once read.
        let (closed, address) = g.listener();
        g.close(closed);
        let failing = g.socket(2, 1 | SOCK_NONBLOCK_K);
        match g.call(cb_kernel_connect, &[failing as u64, address, 16]) {
            Err(EINPROGRESS_) => {
                let entry = g.base + 0x400;
                g.ctx.write_bytes(entry, &[(failing as u32).to_le_bytes().as_slice(), &[5, 0, 0, 0]].concat());
                assert_eq!(fs::poll(&g.ctx, entry, 1, 6000), Ok(1));
                assert_eq!(g.get_int(failing, SOL_SOCKET, SO_ERROR), Ok(ECONNREFUSED_));
                assert_eq!(g.get_int(failing, SOL_SOCKET, SO_ERROR), Ok(0));
            }
            r => assert_eq!(r, Err(ECONNREFUSED_)),
        }
        for fd in [failing, stream, datagram] {
            g.close(fd);
        }
    });
}

/// SO_RCVTIMEO / SO_SNDTIMEO take a timeval, read back as FreeBSD rounds it
/// through its binary time, are EDOM when out of range and EINVAL when
/// short, and bound a blocking receive (EWOULDBLOCK once it passes).
#[test]
fn socket_timeouts_round_and_apply() {
    on_thread(|| {
        let mut g = Guest::new(0x7E76_0000_0000);
        let u = g.socket(2, 2);
        let any = g.loopback(0x40, 0);
        g.call(cb_kernel_bind, &[u as u64, any, 16]).unwrap();
        let (tv, out, len) = (g.base + 0x100, g.base + 0x120, g.base + 0x140);
        let set = |g: &mut Guest, option: i32, sec: i64, usec: i64, size: u64| {
            g.ctx.write_bytes(tv, &[sec.to_le_bytes(), usec.to_le_bytes()].concat());
            g.call(cb_posix_setsockopt, &[u as u64, SOL_SOCKET as u64, option as u64, tv, size])
        };
        let read = |g: &mut Guest, option: i32| {
            g.ctx.write_u32(len, 16);
            g.call(cb_posix_getsockopt, &[u as u64, SOL_SOCKET as u64, option as u64, out, len]).unwrap();
            (g.ctx.read_u64(out).unwrap() as i64, g.ctx.read_u64(out + 8).unwrap() as i64)
        };
        assert_eq!(read(&mut g, SO_RCVTIMEO), (0, 0));
        assert_eq!(set(&mut g, SO_SNDTIMEO, 1, 500_000, 16), Ok(0));
        assert_eq!(read(&mut g, SO_SNDTIMEO), (1, 499_999));
        assert_eq!(set(&mut g, SO_SNDTIMEO, 0, 1, 16), Ok(0));
        assert_eq!(read(&mut g, SO_SNDTIMEO), (0, 0));
        assert_eq!(set(&mut g, SO_SNDTIMEO, 1 << 31, 7, 16), Ok(0));
        assert_eq!(read(&mut g, SO_SNDTIMEO), (0x7FFF_FFFF, 999_999));
        assert_eq!(set(&mut g, SO_RCVTIMEO, -1, 0, 16), Err(EDOM_));
        assert_eq!(set(&mut g, SO_RCVTIMEO, 0, 1_000_000, 16), Err(EDOM_));
        assert_eq!(set(&mut g, SO_RCVTIMEO, 0, 0, 15), Err(EINVAL_));

        assert_eq!(set(&mut g, SO_RCVTIMEO, 0, 60_000, 16), Ok(0));
        let started = std::time::Instant::now();
        assert_eq!(g.call(cb_posix_recvfrom, &[u as u64, g.base, 4, 0, 0, 0]), Err(EWOULDBLOCK_));
        assert!(started.elapsed() >= std::time::Duration::from_millis(59));
        g.close(u);
    });
}

/// fstat on a socket: S_IFSOCK with read and write bits, the queued bytes
/// as the size and the send buffer as the block size; shutting a direction
/// down drops its bits.
#[test]
fn fstat_reports_socket_status() {
    on_thread(|| {
        let mut g = Guest::new(0x7E77_0000_0000);
        let (a, b) = posix_ext::test_socketpair(&g.ctx, 1, g.base);
        let st = g.base + 0x100;
        assert_eq!(fs::fstat(&g.ctx, a, st), Ok(0));
        assert_eq!(g.ctx.read_bytes(st + 8, 2).unwrap(), 0o140666u16.to_le_bytes().to_vec());
        assert_eq!(g.ctx.read_u64(st + 72), Some(0));
        assert!(g.ctx.read_u32(st + 88).unwrap() > 0);
        assert_eq!(kernel_socket_send_bytes(b, b"12345"), Some(Ok(5)));
        assert_eq!(fs::fstat(&g.ctx, a, st), Ok(0));
        assert_eq!(g.ctx.read_u64(st + 72), Some(5));
        g.call(posix_ext::cb_kernel_shutdown, &[a as u64, 1]).unwrap();
        assert_eq!(fs::fstat(&g.ctx, a, st), Ok(0));
        assert_eq!(g.ctx.read_bytes(st + 8, 2).unwrap(), 0o140444u16.to_le_bytes().to_vec());
        g.close(a);
        g.close(b);
        // The number is free again (a test running alongside may reuse it).
        assert!(!is_kernel_socket(a) || fs::fstat(&g.ctx, a, st).is_ok());
    });
}

/// Every host socket error maps to the FreeBSD errno of its name; one
/// without a FreeBSD counterpart is EIO.
#[test]
fn host_errors_map_to_freebsd_numbers() {
    let cases = [
        (libc::EPERM, 1),
        (libc::EINTR, 4),
        (libc::EBADF, 9),
        (libc::ENOMEM, 12),
        (libc::EACCES, 13),
        (libc::EFAULT, 14),
        (libc::EINVAL, 22),
        (libc::ENFILE, 23),
        (libc::EMFILE, 24),
        (libc::EPIPE, 32),
        (libc::EAGAIN, 35),
        (libc::EWOULDBLOCK, 35),
        (libc::EINPROGRESS, 36),
        (libc::EALREADY, 37),
        (libc::ENOTSOCK, 38),
        (libc::EDESTADDRREQ, 39),
        (libc::EMSGSIZE, 40),
        (libc::EPROTOTYPE, 41),
        (libc::ENOPROTOOPT, 42),
        (libc::EPROTONOSUPPORT, 43),
        (libc::ESOCKTNOSUPPORT, 44),
        (libc::EOPNOTSUPP, 45),
        (libc::EPFNOSUPPORT, 46),
        (libc::EAFNOSUPPORT, 47),
        (libc::EADDRINUSE, 48),
        (libc::EADDRNOTAVAIL, 49),
        (libc::ENETDOWN, 50),
        (libc::ENETUNREACH, 51),
        (libc::ENETRESET, 52),
        (libc::ECONNABORTED, 53),
        (libc::ECONNRESET, 54),
        (libc::ENOBUFS, 55),
        (libc::EISCONN, 56),
        (libc::ENOTCONN, 57),
        (libc::ESHUTDOWN, 58),
        (libc::ETOOMANYREFS, 59),
        (libc::ETIMEDOUT, 60),
        (libc::ECONNREFUSED, 61),
        (libc::EHOSTDOWN, 64),
        (libc::EHOSTUNREACH, 65),
        (libc::EPROTO, 92),
    ];
    for (host, guest) in cases {
        assert_eq!(map_errno(host), guest, "host errno {host}");
    }
    assert_eq!(map_errno(libc::ENOENT), EIO_);
    assert_eq!(map_errno(-1), EIO_);
}

/// Sockets take ordinary descriptor numbers from the table files use: the
/// lowest free one, so they stay below FD_SETSIZE however many were
/// opened and closed before.
#[test]
fn sockets_take_low_descriptor_numbers() {
    on_thread(|| {
        let mut g = Guest::new(0x7E78_0000_0000);
        for _ in 0..2000 {
            let fd = g.socket(2, 2);
            assert!((3..1024).contains(&fd), "socket fd {fd}");
            g.close(fd);
        }
    });
}

/// poll() on descriptors that are not sockets, as FreeBSD answers: a
/// negative descriptor is skipped, one that is not open is POLLNVAL,
/// stdin never ready and stdout writable, a file readable and writable, a
/// pipe by its state (a gone reader is a hang-up), a stream socket without
/// a connection never ready; unknown event bits and a timeout below -1 are
/// EINVAL, and a poll of nothing still waits its timeout.
#[test]
fn poll_on_descriptors_that_are_not_sockets() {
    on_thread(|| {
        let mut g = Guest::new(0x7E79_0000_0000);
        let entries = g.base + 0x100;
        let poll = |g: &Guest, fds: &[(i32, i16)], timeout: i32| -> (Result<i64, i32>, Vec<i16>) {
            for (i, &(fd, events)) in fds.iter().enumerate() {
                let mut rec = [0u8; 8];
                rec[..4].copy_from_slice(&fd.to_le_bytes());
                rec[4..6].copy_from_slice(&events.to_le_bytes());
                rec[6..8].copy_from_slice(&0x7777i16.to_le_bytes());
                g.ctx.write_bytes(entries + i as u64 * 8, &rec);
            }
            let r = fs::poll(&g.ctx, entries, fds.len() as u32, timeout);
            (r, (0..fds.len()).map(|i| (g.ctx.read_u32(entries + i as u64 * 8 + 4).unwrap() >> 16) as i16).collect())
        };
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../local/tmp/poll-test-{}", std::process::id()));
        let file = fs::open_host_file(&path);
        let (r, w) = fs::pipe(&g.ctx, g.base + 0x80);
        let idle = g.socket(2, 1);
        let (n, rev) = poll(&g, &[(-1, 1), (0x7FFF_0000, 1), (0, 1), (1, 5), (file, 5), (r, 1), (w, 4), (idle, 5)], 0);
        assert_eq!(n, Ok(4));
        assert_eq!(rev, vec![0, 0x20, 0, 4, 5, 0, 4, 0]);
        g.ctx.write_bytes(g.base + 0x90, b"z");
        assert_eq!(fs::write(&g.ctx, w, g.base + 0x90, 1), Ok(1));
        assert_eq!(poll(&g, &[(r, 1 | 0x40)], 1000), (Ok(1), vec![1 | 0x40]));
        assert_eq!(fs::close(r), Ok(0));
        let (n, rev) = poll(&g, &[(w, 4)], 0);
        assert_eq!(n, Ok(1));
        assert_eq!(rev[0] & 0x14, 0x10, "a pipe without a reader hangs up and is no longer writable");
        assert_eq!(poll(&g, &[(file, 0x200)], 0).0, Err(EINVAL_));
        assert_eq!(fs::poll(&g.ctx, entries, 1, -2), Err(EINVAL_));
        assert_eq!(fs::poll(&g.ctx, 0, 1, 0), Err(EFAULT_));
        let started = std::time::Instant::now();
        assert_eq!(fs::poll(&g.ctx, 0, 0, 30), Ok(0));
        assert!(started.elapsed() >= std::time::Duration::from_millis(25));
        assert_eq!(fs::close(w), Ok(0));
        assert_eq!(fs::close(file), Ok(0));
        g.close(idle);
        let _ = std::fs::remove_file(&path);
    });
}

/// select() on descriptors that are not sockets: files are ready, a pipe
/// by its state, a kqueue without events is not readable, stdout is
/// writable; nfds past FD_SETSIZE is EINVAL and a set descriptor that is
/// not open EBADF.
#[test]
fn select_on_descriptors_that_are_not_sockets() {
    on_thread(|| {
        let mut g = Guest::new(0x7E7A_0000_0000);
        let (rd, wr) = (g.base + 0x100, g.base + 0x200);
        let set = |g: &Guest, at: u64, fds: &[i32]| {
            g.ctx.write_bytes(at, &[0u8; 128]);
            for &fd in fds {
                let word = at + (fd as u64 / 64) * 8;
                let v = g.ctx.read_u64(word).unwrap() | 1 << (fd % 64);
                g.ctx.write_u64(word, v);
            }
        };
        let has = |g: &Guest, at: u64, fd: i32| g.ctx.read_u64(at + (fd as u64 / 64) * 8).unwrap() >> (fd % 64) & 1 != 0;
        let zero = g.base + 0x300;
        g.ctx.write_bytes(zero, &[0u8; 16]);
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../local/tmp/select-test-{}", std::process::id()));
        let file = fs::open_host_file(&path);
        let (r, w) = fs::pipe(&g.ctx, g.base + 0x80);
        let kq = g.call(crate::sync::kqueue::EXPORTS[0].3, &[]).unwrap() as i32;
        let top = [file, r, w, kq, 1].into_iter().max().unwrap() + 1;
        set(&g, rd, &[file, r, kq]);
        set(&g, wr, &[w, 1]);
        assert_eq!(fs::select(&g.ctx, top, [rd, wr, 0], zero), Ok(3));
        assert!(has(&g, rd, file) && !has(&g, rd, r) && !has(&g, rd, kq));
        assert!(has(&g, wr, w) && has(&g, wr, 1));
        g.ctx.write_bytes(g.base + 0x90, b"z");
        assert_eq!(fs::write(&g.ctx, w, g.base + 0x90, 1), Ok(1));
        set(&g, rd, &[r]);
        assert_eq!(fs::select(&g.ctx, top, [rd, 0, 0], zero), Ok(1));
        assert!(has(&g, rd, r));
        assert_eq!(fs::select(&g.ctx, 1025, [rd, 0, 0], zero), Err(EINVAL_));
        set(&g, rd, &[1000]);
        assert_eq!(fs::select(&g.ctx, 1001, [rd, 0, 0], zero), Err(EBADF_));
        for fd in [file, r, w, kq] {
            assert_eq!(fs::close(fd), Ok(0));
        }
        let _ = std::fs::remove_file(&path);
    });
}

/// ioctl and fcntl on descriptors that are not sockets: FIOCLEX/FIONCLEX
/// set FD_CLOEXEC, FIONREAD counts a pipe's queued bytes (0 on its write
/// end), terminal requests are ENOTTY, a request carrying data with a NULL
/// argument is EFAULT before EBADF; F_GETFL reports the open access mode
/// and status flags, which F_SETFL and FIONBIO change and duplicates share.
#[test]
fn ioctl_and_fcntl_on_descriptors_that_are_not_sockets() {
    on_thread(|| {
        let g = Guest::new(0x7E7B_0000_0000);
        let arg = g.base + 0x100;
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../local/tmp/ioctl-test-{}", std::process::id()));
        let file = fs::open_host_file_with(&path, 0x2 | 0x8);
        assert_eq!(fs::fcntl(file, F_GETFL, 0), Ok(0xA));
        assert_eq!(fs::ioctl(&g.ctx, file, 0x2000_6601, 0), Ok(0));
        assert_eq!(fs::fcntl(file, F_GETFD, 0), Ok(1));
        assert_eq!(fs::ioctl(&g.ctx, file, 0x2000_6602, 0), Ok(0));
        assert_eq!(fs::fcntl(file, F_GETFD, 0), Ok(0));
        g.ctx.write_u32(arg, 1);
        assert_eq!(fs::ioctl(&g.ctx, file, 0x8004_667E, arg), Ok(0));
        assert_eq!(fs::fcntl(file, F_GETFL, 0), Ok(0xA | 4));
        let copy = fs::fcntl(file, 0, 0).unwrap() as i32;
        assert_eq!(fs::fcntl(copy, F_GETFL, 0), Ok(0xA | 4));
        assert_eq!(fs::fcntl(copy, F_SETFL, 0), Ok(0));
        assert_eq!(fs::fcntl(file, F_GETFL, 0), Ok(2), "duplicates share the status flags");
        assert_eq!(fs::ioctl(&g.ctx, file, 0x4004_7413, arg), Err(25), "a terminal request");
        assert_eq!(fs::ioctl(&g.ctx, 0x7FFF_0000, 0x8004_667E, 0), Err(EFAULT_));
        assert_eq!(fs::ioctl(&g.ctx, 0x7FFF_0000, 0x4004_667F, arg), Err(EBADF_));

        let (r, w) = fs::pipe(&g.ctx, g.base + 0x80);
        assert_eq!(fs::fcntl(r, F_GETFL, 0), Ok(0));
        assert_eq!(fs::fcntl(w, F_GETFL, 0), Ok(1));
        g.ctx.write_bytes(g.base + 0x90, b"abc");
        assert_eq!(fs::write(&g.ctx, w, g.base + 0x90, 3), Ok(3));
        assert_eq!(fs::ioctl(&g.ctx, r, 0x4004_667F, arg), Ok(0));
        assert_eq!(g.ctx.read_u32(arg), Some(3));
        assert_eq!(fs::ioctl(&g.ctx, w, 0x4004_667F, arg), Ok(0));
        assert_eq!(g.ctx.read_u32(arg), Some(0));
        // A non-blocking empty pipe has nothing to read.
        assert_eq!(fs::fcntl(r, F_SETFL, 4), Ok(0));
        assert_eq!(fs::read(&g.ctx, r, g.base + 0xA0, 8), Ok(3));
        assert_eq!(fs::read(&g.ctx, r, g.base + 0xA0, 8), Err(EWOULDBLOCK_));
        for fd in [r, w, copy, file] {
            assert_eq!(fs::close(fd), Ok(0));
        }
        let _ = std::fs::remove_file(&path);
    });
}
