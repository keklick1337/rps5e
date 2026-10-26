// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libkernel sendmsg / recvmsg / sendfile on kernel sockets, and
//! __inet_ntop / __inet_pton (errno-reporting address conversion).

#[cfg(windows)]
#[allow(unused_imports)]
use crate::host::c as libc;
use super::scenet_ext::{read_message, Message};
use super::*;

const MSGHDR_NAME_LENGTH: u64 = 8;
const MSGHDR_CONTROL_LENGTH: u64 = 40;
const MSGHDR_FLAGS: u64 = 44;
const MSG_TRUNC: u32 = 0x10;
const ENOSPC_: i32 = 28;
const SOCK_STREAM_: i32 = 1;

fn gather(ctx: &CpuContext, m: &Message) -> Result<Vec<u8>, i32> {
    let mut payload = Vec::with_capacity(m.total);
    for &(base, length) in &m.iov {
        if length != 0 {
            payload.extend_from_slice(&ctx.read_bytes(base, length as usize).ok_or(EFAULT_)?);
        }
    }
    Ok(payload)
}

/// sendmsg(fd, const msghdr*, flags): the iovecs gathered into one send,
/// to msg_name as sendto would. Control data is not supported.
fn kernel_sendmsg(ctx: &CpuContext, fd: i32, msg: u64, flags: i32) -> Result<i64, i32> {
    let s = socket_ref(fd)?;
    let m = read_message(ctx, msg)?;
    if m.control != 0 && m.control_length != 0 {
        log::error!("[net] sendmsg: control data ({} bytes) is not supported", m.control_length);
        return Err(EOPNOTSUPP_);
    }
    let payload = gather(ctx, &m)?;
    let host_flags = send_host_flags(flags)?;
    let dest = send_destination(ctx, fd, &s, m.name, if m.name == 0 { 0 } else { m.name_length })?;
    host_send(&s, &payload, host_flags, dest)
}

/// recvmsg(fd, msghdr*, flags): one receive scattered over the iovecs; a
/// longer datagram is cut and MSG_TRUNC set. msg_name gets the sender
/// (inet), the unnamed address (local datagram) or nothing (streams); no
/// control data is returned.
fn kernel_recvmsg(ctx: &CpuContext, fd: i32, msg: u64, flags: i32) -> Result<i64, i32> {
    let s = socket_ref(fd)?;
    let m = read_message(ctx, msg)?;
    let host_flags = receive_host_flags(flags)?;
    let records = s.ty == 2 || s.ty == 5;
    let mut buffer = vec![0u8; m.total + records as usize];
    let (mut received, sender) = host_receive(&s, &mut buffer, host_flags)?;
    let mut out_flags = 0;
    if received > m.total {
        received = m.total;
        out_flags = MSG_TRUNC;
    }
    let mut offset = 0;
    for &(base, length) in &m.iov {
        if offset >= received {
            break;
        }
        let count = (length as usize).min(received - offset);
        if !ctx.write_bytes(base, &buffer[offset..offset + count]) {
            return Err(EFAULT_);
        }
        offset += count;
    }
    if m.name != 0 {
        if s.family == 1 && s.ty != SOCK_STREAM_ {
            posix_ext::write_unnamed_address(ctx, m.name, msg + MSGHDR_NAME_LENGTH)?;
        } else {
            let sender = if s.family == 1 { None } else { sender };
            convert_host_sockaddr(ctx, sender, m.name, msg + MSGHDR_NAME_LENGTH)?;
        }
    }
    if !ctx.write_u32(msg + MSGHDR_FLAGS, out_flags) || !ctx.write_u32(msg + MSGHDR_CONTROL_LENGTH, 0) {
        return Err(EFAULT_);
    }
    Ok(received as i64)
}

fn cb_kernel_sendmsg(ctx: &mut CpuContext) -> HleResult {
    match kernel_sendmsg(ctx, ctx.rdi() as i32, ctx.rsi(), ctx.rdx() as i32) {
        Ok(n) => n,
        Err(e) => posix_socket_failure(ctx, e),
    }
}

fn cb_kernel_recvmsg(ctx: &mut CpuContext) -> HleResult {
    match kernel_recvmsg(ctx, ctx.rdi() as i32, ctx.rsi(), ctx.rdx() as i32) {
        Ok(n) => n,
        Err(e) => posix_socket_failure(ctx, e),
    }
}

/// struct sf_hdtr: headers iovec*, hdr_cnt int, trailers iovec*, trl_cnt.
const SF_HDTR_SIZE: usize = 32;
const SENDFILE_CHUNK: usize = 0x10000;

fn read_iovecs(ctx: &CpuContext, base: u64, count: i32) -> Result<Vec<(u64, u64)>, i32> {
    if count < 0 || count > 1024 || (count > 0 && base == 0) {
        return Err(EINVAL_);
    }
    (0..count as u64)
        .map(|i| {
            let e = ctx.read_bytes(base + i * 16, 16).ok_or(EINVAL_)?;
            Ok((u64::from_le_bytes(e[..8].try_into().unwrap()), u64::from_le_bytes(e[8..].try_into().unwrap())))
        })
        .collect()
}

/// Send all of `data` on a connected stream; adds what went out to `sent`.
fn send_all(raw: i32, data: &[u8], sent: &mut i64) -> Result<(), i32> {
    let mut off = 0;
    while off < data.len() {
        let n = retry(|| unsafe { libc::send(raw, data[off..].as_ptr() as *const libc::c_void, data.len() - off, libc::MSG_NOSIGNAL) })
            .map_err(map_errno)?;
        off += n;
        *sent += n as i64;
    }
    Ok(())
}

fn send_iovecs(ctx: &CpuContext, raw: i32, iov: &[(u64, u64)], sent: &mut i64) -> Result<(), i32> {
    for &(base, length) in iov {
        if length != 0 {
            let data = ctx.read_bytes(base, length as usize).ok_or(EFAULT_)?;
            send_all(raw, &data, sent)?;
        }
    }
    Ok(())
}

/// sendfile(fd, s, offset, nbytes, sf_hdtr*, off_t* sbytes, flags): file
/// bytes from `offset` (nbytes 0: to the end) on connected stream socket
/// `s`, between the optional header and trailer iovecs; *sbytes gets the
/// bytes sent, also on failure.
fn kernel_sendfile(ctx: &CpuContext, file: i32, s: i32, offset: i64, length: u64, hdtr: u64, sbytes: u64) -> HleResult {
    if offset < 0 {
        return posix_socket_failure(ctx, EINVAL_);
    }
    let (headers, trailers) = if hdtr == 0 {
        (Vec::new(), Vec::new())
    } else {
        let Some(h) = ctx.read_bytes(hdtr, SF_HDTR_SIZE) else { return posix_socket_failure(ctx, EFAULT_) };
        let at = |o: usize| u64::from_le_bytes(h[o..o + 8].try_into().unwrap());
        match (read_iovecs(ctx, at(0), at(8) as i32), read_iovecs(ctx, at(16), at(24) as i32)) {
            (Ok(a), Ok(b)) => (a, b),
            _ => return posix_socket_failure(ctx, EINVAL_),
        }
    };
    let mut remaining = if length == 0 { u64::MAX } else { length };
    let mut position = offset as u64;
    let mut sent = 0i64;
    let finish = |ctx: &CpuContext, sent: i64, e: i32| -> HleResult {
        if sbytes != 0 {
            ctx.write_u64(sbytes, sent as u64);
        }
        if e == 0 { 0 } else { posix_socket_failure(ctx, e) }
    };
    match crate::filesystem::descriptor_is_regular(file) {
        // A source that is not a regular file is refused before a byte
        // moves; a closed one fails as the read would.
        Some(false) => return finish(ctx, 0, EINVAL_),
        None => return posix_socket_failure(ctx, EBADF_),
        Some(true) => {}
    }
    let read = |position: u64, remaining: u64| crate::filesystem::pread_bytes(file, (remaining.min(SENDFILE_CHUNK as u64)) as usize, position);
    let mut chunk = match read(position, remaining) {
        Ok(c) => c,
        Err(EBADF_) => return posix_socket_failure(ctx, EBADF_),
        Err(e) => return finish(ctx, 0, if e == 21 || e == 29 { EINVAL_ } else { e }),
    };
    let socket = ksockets().get(&s).map(|k| (k.ty, k.host_handle()));
    let Some((ty, handle)) = socket else { return finish(ctx, 0, not_a_socket(s)) };
    if ty != SOCK_STREAM_ {
        return finish(ctx, 0, EINVAL_);
    }
    let Some(handle) = handle else { return finish(ctx, 0, ENOTCONN_) };
    let raw = handle.fd();
    if let Err(e) = send_iovecs(ctx, raw, &headers, &mut sent) {
        return finish(ctx, sent, e);
    }
    while !chunk.is_empty() {
        if let Err(e) = send_all(raw, &chunk, &mut sent) {
            return finish(ctx, sent, e);
        }
        position += chunk.len() as u64;
        remaining -= chunk.len() as u64;
        if remaining == 0 {
            break;
        }
        chunk = match read(position, remaining) {
            Ok(c) => c,
            Err(e) => return finish(ctx, sent, e),
        };
    }
    if let Err(e) = send_iovecs(ctx, raw, &trailers, &mut sent) {
        return finish(ctx, sent, e);
    }
    finish(ctx, sent, 0)
}

/// (fd, s, offset, nbytes, hdtr, sbytes; flags on the stack, unused).
fn cb_kernel_sendfile(ctx: &mut CpuContext) -> HleResult {
    kernel_sendfile(ctx, ctx.rdi() as i32, ctx.rsi() as i32, ctx.rdx() as i64, ctx.rcx(), ctx.r8(), ctx.r9())
}

/// __inet_pton / inet_pton(af, src, dst) -> 1, 0 for text that is no
/// address, or -1 with errno, in the console's order (measured on a PS5,
/// where both names are one function): the family (EAFNOSUPPORT, even with
/// NULL pointers), the text (NULL: EFAULT), the parse (0, even with a NULL
/// destination), then the destination (NULL: EFAULT).
pub(super) fn cb_inet_pton_errno(ctx: &mut CpuContext) -> HleResult {
    let (family, src, dst) = (ctx.rdi() as i32, ctx.rsi(), ctx.rdx());
    if family != ORBIS_AF_INET && family != ORBIS_AF_INET6 {
        return posix_socket_failure(ctx, EAFNOSUPPORT_);
    }
    if src == 0 {
        return posix_socket_failure(ctx, EFAULT_);
    }
    let Some(text) = ctx.read_cstring(src, 64) else { return posix_socket_failure(ctx, EFAULT_) };
    let bytes: Vec<u8> = if family == ORBIS_AF_INET {
        match text.parse::<Ipv4Addr>() {
            Ok(a) => a.octets().to_vec(),
            Err(_) => return 0,
        }
    } else {
        match text.parse::<Ipv6Addr>() {
            Ok(a) => a.octets().to_vec(),
            Err(_) => return 0,
        }
    };
    if dst == 0 {
        return posix_socket_failure(ctx, EFAULT_);
    }
    if ctx.write_bytes(dst, &bytes) { 1 } else { posix_socket_failure(ctx, EFAULT_) }
}

/// __inet_ntop / inet_ntop(af, src, dst, size) -> dst, or NULL with errno,
/// in the console's order (measured on a PS5): the family (EAFNOSUPPORT),
/// the source (NULL: EFAULT), the room for the text and its terminator
/// (ENOSPC), then the destination (NULL: EFAULT). An AF_INET6 text is
/// followed by zero bytes up to `size`; an AF_INET one is not.
pub(super) fn cb_inet_ntop_errno(ctx: &mut CpuContext) -> HleResult {
    let (family, src, dst, size) = (ctx.rdi() as i32, ctx.rsi(), ctx.rdx(), ctx.rcx() as u32);
    let len = match family {
        ORBIS_AF_INET => 4,
        ORBIS_AF_INET6 => 16,
        _ => {
            posix_socket_failure(ctx, EAFNOSUPPORT_);
            return 0;
        }
    };
    if src == 0 {
        posix_socket_failure(ctx, EFAULT_);
        return 0;
    }
    let Some(text) = ctx.read_bytes(src, len).and_then(|raw| format_address(family, &raw)) else {
        posix_socket_failure(ctx, EFAULT_);
        return 0;
    };
    if text.len() + 1 > size as usize {
        posix_socket_failure(ctx, ENOSPC_);
        return 0;
    }
    if dst == 0 {
        posix_socket_failure(ctx, EFAULT_);
        return 0;
    }
    let mut out = text.into_bytes();
    out.push(0);
    if family == ORBIS_AF_INET6 {
        out.resize(size as usize, 0);
    }
    if ctx.write_bytes(dst, &out) {
        dst as HleResult
    } else {
        posix_socket_failure(ctx, EFAULT_);
        0
    }
}

pub(super) const KMSG_EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("aNeavPDNKzA", "sendmsg", cb_kernel_sendmsg),
    ("hI7oVeOluPM", "recvmsg", cb_kernel_recvmsg),
    ("YA0r4LCkfeY", "sendfile", cb_kernel_sendfile),
    ("fyPeCKJ94Hg", "__inet_pton", cb_inet_pton_errno),
    ("4pYihoPggn8", "__inet_ntop", cb_inet_ntop_errno),
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CpuRegister;
    use ps5vm::{MemProtect, VirtualMemory};

    const BASE: u64 = 0x7E64_0000_0000;

    fn call(ctx: &mut CpuContext, cb: HleCallback, args: &[u64]) -> HleResult {
        let regs = [CpuRegister::Rdi, CpuRegister::Rsi, CpuRegister::Rdx, CpuRegister::Rcx, CpuRegister::R8, CpuRegister::R9];
        for (r, v) in regs.iter().zip(args.iter().chain(std::iter::repeat(&0))) {
            ctx.set(*r, *v);
        }
        cb(ctx)
    }

    /// msghdr at `msg` with `iov` iovecs at `iov_at`; name and control as given.
    fn msghdr(ctx: &CpuContext, msg: u64, name: u64, namelen: u32, iov_at: u64, iov: &[(u64, u64)], control: u64, controllen: u32) {
        let mut b = vec![0u8; 48];
        b[0..8].copy_from_slice(&name.to_le_bytes());
        b[8..12].copy_from_slice(&namelen.to_le_bytes());
        b[16..24].copy_from_slice(&iov_at.to_le_bytes());
        b[24..28].copy_from_slice(&(iov.len() as u32).to_le_bytes());
        b[32..40].copy_from_slice(&control.to_le_bytes());
        b[40..44].copy_from_slice(&controllen.to_le_bytes());
        ctx.write_bytes(msg, &b);
        for (i, (base, len)) in iov.iter().enumerate() {
            ctx.write_u64(iov_at + i as u64 * 16, *base);
            ctx.write_u64(iov_at + i as u64 * 16 + 8, *len);
        }
    }

    #[test]
    fn sendmsg_recvmsg_sendfile_and_inet() {
        std::thread::spawn(|| {
            let vm: &'static VirtualMemory = Box::leak(Box::new(VirtualMemory::new()));
            vm.map(BASE, 0x10000, MemProtect::RW).unwrap();
            let mut ctx = CpuContext::new(vm);
            let errno = crate::kernel::bind_test_errno(&ctx);
            let err = |ctx: &CpuContext| ctx.read_u32(errno).unwrap() as i32;
            let (msg, iov, data, out) = (BASE + 0x100, BASE + 0x200, BASE + 0x400, BASE + 0x800);

            // Stream pair: gather on send, scatter on receive.
            let (a, b) = posix_ext::test_socketpair(&ctx, 1, BASE);
            ctx.write_bytes(data, b"hello world");
            msghdr(&ctx, msg, 0, 0, iov, &[(data, 5), (data + 5, 6)], 0, 0);
            assert_eq!(call(&mut ctx, cb_kernel_sendmsg, &[a as u64, msg, 0]), 11);
            msghdr(&ctx, msg, data + 0x100, 16, iov, &[(out, 4), (out + 0x10, 16)], 0, 0);
            assert_eq!(call(&mut ctx, cb_kernel_recvmsg, &[b as u64, msg, 0]), 11);
            assert_eq!(ctx.read_bytes(out, 4).unwrap(), b"hell");
            assert_eq!(ctx.read_bytes(out + 0x10, 7).unwrap(), b"o world");
            assert_eq!(ctx.read_u32(msg + MSGHDR_NAME_LENGTH), Some(0), "a local stream has no peer name");
            assert_eq!(ctx.read_u32(msg + MSGHDR_FLAGS), Some(0));
            // Control data and unknown flags are refused; a file is not a socket.
            msghdr(&ctx, msg, 0, 0, iov, &[(data, 5)], data, 16);
            assert_eq!(call(&mut ctx, cb_kernel_sendmsg, &[a as u64, msg, 0]), -1);
            assert_eq!(err(&ctx), EOPNOTSUPP_);
            msghdr(&ctx, msg, 0, 0, iov, &[(data, 5)], 0, 0);
            assert_eq!(call(&mut ctx, cb_kernel_sendmsg, &[a as u64, msg, 0x1]), -1);
            assert_eq!(err(&ctx), EOPNOTSUPP_);
            assert_eq!(call(&mut ctx, cb_kernel_sendmsg, &[0x0FFF_FFF0, msg, 0]), -1);
            assert_eq!(err(&ctx), EBADF_);
            assert_eq!(call(&mut ctx, cb_kernel_recvmsg, &[b as u64, 0, 0]), -1);
            assert_eq!(err(&ctx), EFAULT_);

            // sendfile with a header and trailer.
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../local/tmp/sendfile-test-{}", std::process::id()));
            std::fs::write(&path, b"0123456789").unwrap();
            let file = crate::filesystem::test_hooks::open_host_file(&path);
            ctx.write_bytes(data, b"HDR");
            ctx.write_bytes(data + 0x10, b"TRL");
            let (hiov, tiov, hdtr, sbytes) = (BASE + 0x300, BASE + 0x320, BASE + 0x340, BASE + 0x380);
            ctx.write_u64(hiov, data);
            ctx.write_u64(hiov + 8, 3);
            ctx.write_u64(tiov, data + 0x10);
            ctx.write_u64(tiov + 8, 3);
            for (o, v) in [(0u64, hiov), (8, 1), (16, tiov), (24, 1)] {
                ctx.write_u64(hdtr + o, v);
            }
            assert_eq!(call(&mut ctx, cb_kernel_sendfile, &[file as u64, a as u64, 2, 5, hdtr, sbytes]), 0);
            assert_eq!(ctx.read_u64(sbytes), Some(11));
            assert_eq!(recv_bytes(b, 64), b"HDR23456TRL");
            assert_eq!(call(&mut ctx, cb_kernel_sendfile, &[file as u64, a as u64, 7, 0, 0, sbytes]), 0);
            assert_eq!(recv_bytes(b, 64), b"789");
            assert_eq!(call(&mut ctx, cb_kernel_sendfile, &[file as u64, a as u64, u64::MAX, 0, 0, 0]), -1);
            assert_eq!(err(&ctx), EINVAL_);
            assert_eq!(call(&mut ctx, cb_kernel_sendfile, &[0x0FFF_FFF0, a as u64, 0, 0, 0, 0]), -1);
            assert_eq!(err(&ctx), EBADF_);
            assert_eq!(call(&mut ctx, cb_kernel_sendfile, &[file as u64, file as u64, 0, 0, 0, sbytes]), -1);
            assert_eq!(err(&ctx), ENOTSOCK_);
            // A source that is not a regular file is refused before a byte
            // moves: /dev/urandom is a character device.
            let random = crate::filesystem::test_hooks::open_host_file(std::path::Path::new("/dev/urandom"));
            ctx.write_u64(sbytes, 77);
            assert_eq!(call(&mut ctx, cb_kernel_sendfile, &[random as u64, a as u64, 0, 16, 0, sbytes]), -1);
            assert_eq!(err(&ctx), EINVAL_);
            assert_eq!(ctx.read_u64(sbytes), Some(0));
            assert_eq!(crate::filesystem::test_hooks::close(random), Ok(0));
            assert_eq!(crate::filesystem::test_hooks::close(file), Ok(0));
            std::fs::remove_file(&path).unwrap();
            assert!(kernel_socket_close(a) && kernel_socket_close(b));

            // Datagram pair: a longer datagram is cut with MSG_TRUNC.
            let (c, d) = posix_ext::test_socketpair(&ctx, 2, BASE);
            ctx.write_bytes(data, b"datagram");
            msghdr(&ctx, msg, 0, 0, iov, &[(data, 8)], 0, 0);
            assert_eq!(call(&mut ctx, cb_kernel_sendmsg, &[c as u64, msg, 0]), 8);
            msghdr(&ctx, msg, 0, 0, iov, &[(out, 4)], 0, 0);
            assert_eq!(call(&mut ctx, cb_kernel_recvmsg, &[d as u64, msg, 0]), 4);
            assert_eq!(ctx.read_u32(msg + MSGHDR_FLAGS), Some(MSG_TRUNC));
            assert_eq!(ctx.read_bytes(out, 4).unwrap(), b"data");
            assert!(kernel_socket_close(c) && kernel_socket_close(d));

            // __inet_pton / __inet_ntop.
            let (text, bin) = (BASE + 0x900, BASE + 0x980);
            ctx.write_bytes(text, b"192.168.1.20\0");
            assert_eq!(call(&mut ctx, cb_inet_pton_errno, &[2, text, bin]), 1);
            assert_eq!(ctx.read_bytes(bin, 4).unwrap(), vec![192, 168, 1, 20]);
            ctx.write_bytes(text, b"1.2.3\0");
            assert_eq!(call(&mut ctx, cb_inet_pton_errno, &[2, text, bin]), 0);
            ctx.write_bytes(text, b"fe80::1\0");
            assert_eq!(call(&mut ctx, cb_inet_pton_errno, &[28, text, bin]), 1);
            assert_eq!(ctx.read_bytes(bin, 2).unwrap(), vec![0xfe, 0x80]);
            assert_eq!(call(&mut ctx, cb_inet_pton_errno, &[5, text, bin]), -1);
            assert_eq!(err(&ctx), EAFNOSUPPORT_);
            assert_eq!(call(&mut ctx, cb_inet_pton_errno, &[2, 0, bin]), -1);
            assert_eq!(err(&ctx), EFAULT_);
            ctx.write_bytes(bin, &[10, 0, 0, 1]);
            assert_eq!(call(&mut ctx, cb_inet_ntop_errno, &[2, bin, text, 16]), text as HleResult);
            assert_eq!(ctx.read_cstring(text, 16).as_deref(), Some("10.0.0.1"));
            assert_eq!(call(&mut ctx, cb_inet_ntop_errno, &[2, bin, text, 8]), 0);
            assert_eq!(err(&ctx), ENOSPC_);
            assert_eq!(call(&mut ctx, cb_inet_ntop_errno, &[9, bin, text, 16]), 0);
            assert_eq!(err(&ctx), EAFNOSUPPORT_);
            // Measured on a PS5: the family before NULL pointers, the parse
            // and the room before the destination, IPv6 zero padding.
            assert_eq!(call(&mut ctx, cb_inet_pton_errno, &[5, 0, 0]), -1);
            assert_eq!(err(&ctx), EAFNOSUPPORT_);
            ctx.write_bytes(text, b"1.2.3.04\0");
            assert_eq!(call(&mut ctx, cb_inet_pton_errno, &[2, text, 0]), 0);
            ctx.write_bytes(text, b"1.2.3.4\0");
            assert_eq!(call(&mut ctx, cb_inet_pton_errno, &[2, text, 0]), -1);
            assert_eq!(err(&ctx), EFAULT_);
            assert_eq!(call(&mut ctx, cb_inet_ntop_errno, &[2, bin, 0, 4]), 0);
            assert_eq!(err(&ctx), ENOSPC_);
            assert_eq!(call(&mut ctx, cb_inet_ntop_errno, &[2, bin, 0, 16]), 0);
            assert_eq!(err(&ctx), EFAULT_);
            ctx.write_bytes(text, &[0xAA; 32]);
            assert_eq!(call(&mut ctx, cb_inet_ntop_errno, &[2, bin, text, 16]), text as HleResult);
            assert_eq!(ctx.read_bytes(text + 9, 2).unwrap(), vec![0xAA; 2], "AF_INET: nothing past the NUL");
            ctx.write_bytes(bin, &Ipv6Addr::LOCALHOST.octets());
            assert_eq!(call(&mut ctx, cb_inet_ntop_errno, &[28, bin, text, 16]), text as HleResult);
            assert_eq!(ctx.read_bytes(text, 17).unwrap(), [&b"::1"[..], &[0; 13], &[0xAA]].concat(), "AF_INET6: zeros up to size");
        })
        .join()
        .unwrap();
    }

    /// Everything queued on `fd` (non-blocking reads until it runs dry).
    fn recv_bytes(fd: i32, max: usize) -> Vec<u8> {
        let raw = ksockets().get(&fd).unwrap().host_handle().unwrap().fd();
        let mut all = Vec::new();
        let mut buf = vec![0u8; max];
        loop {
            // SAFETY: a local buffer.
            let n = unsafe { libc::recv(raw, buf.as_mut_ptr() as *mut libc::c_void, buf.len(), libc::MSG_DONTWAIT) };
            if n <= 0 {
                return all;
            }
            all.extend_from_slice(&buf[..n as usize]);
        }
    }

    #[test]
    fn nids_match_names() {
        for &(nid, name, _) in KMSG_EXPORTS {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }
}
