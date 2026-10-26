// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceNet: scatter/gather messages, peer names, 64-bit byte order, the
//! IPv6 resolver call and the diagnostic dumps.

#[cfg(windows)]
#[allow(unused_imports)]
use crate::host::c as libc;
use super::*;

/// SceNetMsghdr: name, namelen (u32), iov, iovlen (int), control,
/// controllen (u32), flags (int); 48 bytes.
const MSGHDR_SIZE: usize = 48;
const MSGHDR_NAME_LENGTH: u64 = 8;
const MSGHDR_CONTROL_LENGTH: u64 = 40;
const MSGHDR_FLAGS: u64 = 44;
/// SceNetIovec: base, length; 16 bytes.
const IOVEC_SIZE: u64 = 16;
/// Most iovecs one message may carry.
const UIO_MAXIOV: i32 = 1024;
const MSG_TRUNC: u32 = 0x10;

/// A guest SceNetMsghdr with its validated iovecs.
pub(super) struct Message {
    pub(super) name: u64,
    pub(super) name_length: u32,
    pub(super) control: u64,
    pub(super) control_length: u32,
    pub(super) iov: Vec<(u64, u64)>,
    pub(super) total: usize,
}

/// Read and check a message header: EFAULT for a NULL header, iovec
/// array or buffer, EMSGSIZE for a bad iovec count, EINVAL when the total
/// length exceeds INT_MAX.
pub(super) fn read_message(ctx: &CpuContext, msg: u64) -> Result<Message, i32> {
    if msg == 0 {
        return Err(EFAULT_);
    }
    let raw = ctx.read_bytes(msg, MSGHDR_SIZE).ok_or(EFAULT_)?;
    let u64_at = |o: usize| u64::from_le_bytes(raw[o..o + 8].try_into().unwrap());
    let u32_at = |o: usize| u32::from_le_bytes(raw[o..o + 4].try_into().unwrap());
    let iov_addr = u64_at(16);
    let iov_length = u32_at(24) as i32;
    if !(0..=UIO_MAXIOV).contains(&iov_length) {
        return Err(EMSGSIZE_);
    }
    if iov_length != 0 && iov_addr == 0 {
        return Err(EFAULT_);
    }
    let mut iov = Vec::with_capacity(iov_length as usize);
    let mut total: u64 = 0;
    for i in 0..iov_length as u64 {
        let entry = ctx.read_bytes(iov_addr + i * IOVEC_SIZE, IOVEC_SIZE as usize).ok_or(EFAULT_)?;
        let base = u64::from_le_bytes(entry[0..8].try_into().unwrap());
        let length = u64::from_le_bytes(entry[8..16].try_into().unwrap());
        if base == 0 && length != 0 {
            return Err(EFAULT_);
        }
        if length > i32::MAX as u64 - total {
            return Err(EINVAL_);
        }
        total += length;
        iov.push((base, length));
    }
    Ok(Message {
        name: u64_at(0),
        name_length: u32_at(8),
        control: u64_at(32),
        control_length: u32_at(40),
        iov,
        total: total as usize,
    })
}

/// sceNetSendmsg(s, const SceNetMsghdr* msg, flags): the iovecs are
/// gathered and sent as one sceNetSendto to msg->name.
fn net_sendmsg(ctx: &CpuContext, s: i32, msg: u64, flags: i32) -> Result<i64, i32> {
    let m = read_message(ctx, msg)?;
    if m.control != 0 && m.control_length != 0 {
        log::warn!("[net] sceNetSendmsg: control data ({} bytes) is not supported", m.control_length);
        return Err(EOPNOTSUPP_);
    }
    let mut payload = Vec::with_capacity(m.total);
    for &(base, length) in &m.iov {
        if length != 0 {
            payload.extend_from_slice(&ctx.read_bytes(base, length as usize).ok_or(EFAULT_)?);
        }
    }
    let host_flags = message_flags(flags | MSG_NOSIGNAL)?;
    let sock = native_socket(s)?;
    let target = if m.name == 0 {
        None
    } else {
        let (ip, port) = convert_guest_sockaddr(ctx, m.name, m.name_length)?;
        Some(sockaddr_from(&outbound_destination(IpAddr::V4(ip), port)?))
    };
    let sent = retry(|| unsafe {
        match &target {
            None => libc::send(sock.fd, payload.as_ptr() as *const libc::c_void, payload.len(), host_flags),
            Some((ss, len)) => libc::sendto(
                sock.fd,
                payload.as_ptr() as *const libc::c_void,
                payload.len(),
                host_flags,
                ss as *const _ as *const libc::sockaddr,
                *len,
            ),
        }
    })
    .map_err(map_errno)?;
    Ok(sent as i64)
}

/// sceNetRecvmsg(s, SceNetMsghdr* msg, flags): one receive scattered over
/// the iovecs. A datagram longer than the buffers is cut and MSG_TRUNC set;
/// the sender's address goes to msg->name, and no control data is returned.
fn net_recvmsg(ctx: &CpuContext, s: i32, msg: u64, flags: i32) -> Result<i64, i32> {
    let m = read_message(ctx, msg)?;
    if flags & !(MSG_PEEK | MSG_DONTWAIT) != 0 {
        return Err(EOPNOTSUPP_);
    }
    let host_flags = message_flags(flags)?;
    let sock = native_socket(s)?;
    // The header and every buffer are checked before anything is received,
    // so a bad pointer leaves the data queued.
    if !ps5vm::ranges::accessible(msg, MSGHDR_SIZE as u64, true)
        || m.iov.iter().any(|&(base, length)| length != 0 && !ps5vm::ranges::accessible(base, length, true))
    {
        return Err(EFAULT_);
    }
    let datagram = get_int_opt(sock.fd, libc::SOL_SOCKET, libc::SO_TYPE).map_err(map_errno)? != libc::SOCK_STREAM;
    let mut buffer = vec![0u8; m.total + datagram as usize];
    let mut ss: libc::sockaddr_storage = unsafe { std::mem::zeroed() };
    let mut sslen = std::mem::size_of::<libc::sockaddr_storage>() as libc::socklen_t;
    let mut received = retry(|| unsafe {
        sslen = std::mem::size_of::<libc::sockaddr_storage>() as libc::socklen_t;
        libc::recvfrom(
            sock.fd,
            buffer.as_mut_ptr() as *mut libc::c_void,
            buffer.len(),
            host_flags,
            &mut ss as *mut _ as *mut libc::sockaddr,
            &mut sslen,
        )
    })
    .map_err(map_errno)?;
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
    if !ctx.write_u32(msg + MSGHDR_FLAGS, out_flags) || !ctx.write_u32(msg + MSGHDR_CONTROL_LENGTH, 0) {
        return Err(EFAULT_);
    }
    if m.name != 0 {
        let peer = sockaddr_to(&ss, sslen);
        if convert_host_sockaddr(ctx, peer, m.name, msg + MSGHDR_NAME_LENGTH).is_err() {
            ctx.write_u32(msg + MSGHDR_NAME_LENGTH, 0);
        }
    }
    Ok(received as i64)
}

fn cb_sendmsg(ctx: &mut CpuContext) -> HleResult {
    let r = net_sendmsg(ctx, ctx.rdi() as i32, ctx.rsi(), ctx.rdx() as i32);
    socket_result(ctx, r)
}

fn cb_recvmsg(ctx: &mut CpuContext) -> HleResult {
    let r = net_recvmsg(ctx, ctx.rdi() as i32, ctx.rsi(), ctx.rdx() as i32);
    socket_result(ctx, r)
}

/// sceNetGetpeername(s, addr, addrlen*): as much of the peer's address as
/// `*addrlen` holds, with `*addrlen` set to the bytes copied.
fn net_getpeername(ctx: &CpuContext, s: i32, addr: u64, len_addr: u64) -> Result<(), i32> {
    let slot = socket_slot(s)?;
    if addr == 0 || len_addr == 0 {
        return Err(EINVAL_);
    }
    let mut ss: libc::sockaddr_storage = unsafe { std::mem::zeroed() };
    let mut len = std::mem::size_of::<libc::sockaddr_storage>() as libc::socklen_t;
    let r = unsafe { libc::getpeername(slot.transport.fd, &mut ss as *mut _ as *mut libc::sockaddr, &mut len) };
    if r < 0 {
        return Err(map_errno(last_errno()));
    }
    let peer = sockaddr_to(&ss, len).ok_or(EAFNOSUPPORT_)?;
    convert_host_sockaddr(ctx, Some(peer), addr, len_addr).map(|_| ())
}

fn cb_getpeername(ctx: &mut CpuContext) -> HleResult {
    let r = net_getpeername(ctx, ctx.rdi() as i32, ctx.rsi(), ctx.rdx());
    socket_result(ctx, r.map(|_| 0))
}

/// sceNetHtonll / sceNetNtohll(uint64_t)
fn cb_htonll(ctx: &mut CpuContext) -> HleResult {
    ctx.rdi().swap_bytes() as i64
}

/// sceNetResolverStartNtoa6(rid, hostname, SceNetIn6Addr* addr, timeout,
/// retry, flags): IPv6 literals resolve locally; the console is offline,
/// so names (and IPv4 literals, which have no IPv6 address) fail with "no
/// DNS". A success clears the resolver's error.
fn cb_resolver_start_ntoa6(ctx: &mut CpuContext) -> HleResult {
    let (id, host_addr, dst, flags) = (ctx.rdi() as i32, ctx.rsi(), ctx.rdx(), ctx.r9() as i32);
    if flags != 0 {
        log::warn!("[net] sceNetResolverStartNtoa6: flags {flags:#x} are not supported");
        return finish(ctx, net_error(EINVAL_));
    }
    if host_addr == 0 || dst == 0 {
        return finish(ctx, net_error(EINVAL_));
    }
    if net().resolver_mut(id).is_none() {
        return finish(ctx, net_error(EBADF_));
    }
    let host = ctx.read_cstr_bytes(host_addr, MAX_NAME_LENGTH).unwrap_or_default();
    if let Some(octets) = inet_pton(ORBIS_AF_INET6, &host) {
        if !ctx.write_bytes(dst, &octets) {
            return finish(ctx, net_error(EFAULT_));
        }
        set_resolver_error(id, 0);
        return 0;
    }
    set_resolver_error(id, NET_ERROR_RESOLVER_NO_DNS);
    log::info!("[net] resolver: '{}' (IPv6) -> no DNS (offline)", String::from_utf8_lossy(&host));
    finish(ctx, NET_ERROR_RESOLVER_NO_DNS)
}

/// sceNetShowIfconfig / sceNetShowNetstat: diagnostic dumps to the
/// console's log; there is nothing to report.
fn cb_show(_ctx: &mut CpuContext) -> HleResult {
    0
}

/// The netstat dump for pool `memid`: a summary line, then one line per
/// socket. None when the pool does not exist.
fn netstat_lines(memid: i32) -> Option<Vec<String>> {
    let (size, sockets) = {
        let st = net();
        if !st.pool_valid(memid) {
            return None;
        }
        let size = st.pools[memid as usize].as_ref().map_or(0, |p| p.size);
        let mut sockets: Vec<(i32, Arc<HostSocket>)> = st.sockets.iter().map(|(&id, s)| (id, s.transport.clone())).collect();
        sockets.sort_by_key(|s| s.0);
        (size, sockets)
    };
    let mut lines = vec![format!("sceNet netstat: pool={memid} bytes={size} sockets={}", sockets.len())];
    for (id, sock) in sockets {
        let local = local_addr(sock.fd);
        let family = match local {
            Some(SocketAddr::V6(_)) => ORBIS_AF_INET6,
            _ => ORBIS_AF_INET,
        };
        let ty = get_int_opt(sock.fd, libc::SOL_SOCKET, libc::SO_TYPE).unwrap_or(0);
        let bound = local.is_some_and(|a| a.port() != 0);
        let listening = get_int_opt(sock.fd, libc::SOL_SOCKET, libc::SO_ACCEPTCONN).unwrap_or(0) != 0;
        lines.push(format!(
            "socket={id} family={family} type={ty} nonblock={} bound={} listening={}",
            !sock.blocking() as i32,
            bound as i32,
            listening as i32
        ));
    }
    Some(lines)
}

/// sceNetShowNetstatWithMemory(memid): dump the sockets to the log.
fn cb_show_netstat_with_memory(ctx: &mut CpuContext) -> HleResult {
    match netstat_lines(ctx.rdi() as i32) {
        Some(lines) => {
            for line in lines {
                log::info!("[net] {line}");
            }
            0
        }
        None => finish(ctx, net_error(EBADF_)),
    }
}

/// sceNetResolverStartAton6(rid, const SceNetIn6Addr* addr, hostname, len,
/// timeout, retry, flags): reverse lookups need DNS, which the offline
/// console does not have.
fn cb_resolver_start_aton6(ctx: &mut CpuContext) -> HleResult {
    let (id, addr, hostname, len, flags) = (ctx.rdi() as i32, ctx.rsi(), ctx.rdx(), ctx.rcx() as i32, ctx.r9() as i32);
    if flags != 0 {
        log::warn!("[net] sceNetResolverStartAton6: flags {flags:#x} are not supported");
        return finish(ctx, net_error(EINVAL_));
    }
    if addr == 0 || hostname == 0 || len <= 0 {
        return finish(ctx, net_error(EINVAL_));
    }
    if net().resolver_mut(id).is_none() {
        return finish(ctx, net_error(EBADF_));
    }
    let Some(raw) = ctx.read_bytes(addr, 16) else {
        return finish(ctx, net_error(EFAULT_));
    };
    set_resolver_error(id, NET_ERROR_RESOLVER_NO_DNS);
    let octets: [u8; 16] = raw.try_into().unwrap();
    log::info!("[net] resolver: reverse {} -> no DNS (offline)", Ipv6Addr::from(octets));
    finish(ctx, NET_ERROR_RESOLVER_NO_DNS)
}

/// sceNetClearDnsCache(flags): there is no library DNS cache to drop.
fn cb_clear_dns_cache(ctx: &mut CpuContext) -> HleResult {
    let flags = ctx.rdi() as i32;
    if flags != 0 {
        log::warn!("[net] sceNetClearDnsCache: flags {flags:#x} are not supported");
        return net_fail(ctx, EINVAL_);
    }
    0
}

/// DNS servers the title set with sceNetSetDnsInfo / sceNetSetDns6Info:
/// (IPv4 pair if set, IPv6 pair).
static DNS_OVERRIDE: std::sync::Mutex<(Option<[u8; 8]>, [u8; 32])> = std::sync::Mutex::new((None, [0; 32]));

/// Servers in a pair: 0 without a primary, 1 without a secondary, else 2.
fn dns_count(pair: &[u8]) -> i64 {
    let (primary, secondary) = pair.split_at(pair.len() / 2);
    if primary.iter().all(|&b| b == 0) {
        0
    } else if secondary.iter().all(|&b| b == 0) {
        1
    } else {
        2
    }
}

/// sceNetGetDnsInfo(SceNetDnsInfo* info, flags) -> server count: two IPv4
/// server slots; the offline console has no DNS server, so both are zero
/// and the count is 0, unless the title set servers with sceNetSetDnsInfo.
fn cb_get_dns_info(ctx: &mut CpuContext) -> HleResult {
    let (info, flags) = (ctx.rdi(), ctx.rsi() as i32);
    if flags != 0 || info == 0 {
        if flags != 0 {
            log::warn!("[net] sceNetGetDnsInfo: flags {flags:#x} are not supported");
        }
        return net_fail(ctx, EINVAL_);
    }
    let servers = DNS_OVERRIDE.lock().unwrap_or_else(|e| e.into_inner()).0.unwrap_or([0; 8]);
    if !ctx.write_bytes(info, &servers) {
        return net_fail(ctx, EFAULT_);
    }
    dns_count(&servers)
}

/// sceNetSetDnsInfo(const SceNetDnsInfo* info, flags): replaces the IPv4
/// servers; NULL goes back to the system's (none). Flags and a zero
/// primary are EINVAL.
fn cb_set_dns_info(ctx: &mut CpuContext) -> HleResult {
    let (info, flags) = (ctx.rdi(), ctx.rsi() as i32);
    if flags != 0 {
        return net_fail(ctx, EINVAL_);
    }
    let servers = if info == 0 {
        None
    } else {
        let Some(b) = ctx.read_bytes(info, 8) else { return net_fail(ctx, EFAULT_) };
        if b[..4].iter().all(|&x| x == 0) {
            return net_fail(ctx, EINVAL_);
        }
        Some(<[u8; 8]>::try_from(&b[..]).unwrap())
    };
    DNS_OVERRIDE.lock().unwrap_or_else(|e| e.into_inner()).0 = servers;
    0
}

/// sceNetSetDns6Info(const SceNetDns6Info* info, flags): the IPv6 pair;
/// NULL clears it.
fn cb_set_dns6_info(ctx: &mut CpuContext) -> HleResult {
    let (info, flags) = (ctx.rdi(), ctx.rsi() as i32);
    if flags != 0 {
        return net_fail(ctx, EINVAL_);
    }
    let mut servers = [0u8; 32];
    if info != 0 {
        let Some(b) = ctx.read_bytes(info, 32) else { return net_fail(ctx, EFAULT_) };
        if b[..16].iter().all(|&x| x == 0) {
            return net_fail(ctx, EINVAL_);
        }
        servers.copy_from_slice(&b);
    }
    DNS_OVERRIDE.lock().unwrap_or_else(|e| e.into_inner()).1 = servers;
    0
}

/// sceNetGetDns6Info(SceNetDns6Info* info, flags) -> server count.
fn cb_get_dns6_info(ctx: &mut CpuContext) -> HleResult {
    let (info, flags) = (ctx.rdi(), ctx.rsi() as i32);
    if flags != 0 || info == 0 {
        return net_fail(ctx, EINVAL_);
    }
    let servers = DNS_OVERRIDE.lock().unwrap_or_else(|e| e.into_inner()).1;
    if !ctx.write_bytes(info, &servers) {
        return net_fail(ctx, EFAULT_);
    }
    dns_count(&servers)
}

/// sceNetGetSystemTime(u64* out): microseconds since the process started.
fn cb_get_system_time(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    if out != 0 {
        ctx.write_u64(out, crate::time::process_time_usec());
    }
    0
}

/// sceNetGetRandom(u32* out): a random 32-bit value.
fn cb_get_random(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    if out == 0 {
        return net_fail(ctx, EINVAL_);
    }
    let mut v = [0u8; 4];
    crate::misc_services::fill_random(&mut v);
    if ctx.write_bytes(out, &v) { 0 } else { net_fail(ctx, EFAULT_) }
}

/// The interface names sceNetGetIfName knows, by index.
const INTERFACE_NAMES: [&str; 10] = ["lo0", "eth0", "eth1", "dbg0", "wlan0", "wlan1", "gbe0", "bt0", "phone0", "pppoe0"];

/// Guest copies of the interface names, then an empty string.
static INTERFACE_NAME_STRINGS: std::sync::Mutex<Option<u64>> = std::sync::Mutex::new(None);

/// sceNetGetIfName(index) -> const char*: the interface's name; an index
/// past the table sets EINVAL and returns an empty string.
fn cb_get_if_name(ctx: &mut CpuContext) -> HleResult {
    const SLOT: u64 = 8;
    let index = ctx.rdi() as u32;
    let base = {
        let mut strings = INTERFACE_NAME_STRINGS.lock().unwrap();
        if strings.is_none() {
            let Some(a) = crate::font::galloc(ctx, SLOT * (INTERFACE_NAMES.len() as u64 + 1)) else { return 0 };
            for (i, name) in INTERFACE_NAMES.iter().enumerate() {
                ctx.write_bytes(a + i as u64 * SLOT, name.as_bytes());
            }
            *strings = Some(a);
        }
        strings.unwrap()
    };
    if index as usize >= INTERFACE_NAMES.len() {
        set_net_error(ctx, 0, EINVAL_);
        return (base + SLOT * INTERFACE_NAMES.len() as u64) as i64;
    }
    (base + SLOT * index as u64) as i64
}

/// sceNetMemoryAllocate(size, flags) -> void*: a block from the guest
/// heap; flag 2 zeroes it (every block here is zeroed). NULL for a size of
/// zero or less, or for flags other than 2.
fn cb_memory_allocate(ctx: &mut CpuContext) -> HleResult {
    let (size, flags) = (ctx.rdi() as i64, ctx.rsi() as i32);
    if flags & !2 != 0 {
        log::warn!("[net] sceNetMemoryAllocate: flags {flags:#x} are not supported");
        return 0;
    }
    if size <= 0 {
        return 0;
    }
    crate::font::galloc(ctx, size as u64).unwrap_or(0) as i64
}

/// sceNetMemoryFree(void*)
fn cb_memory_free(ctx: &mut CpuContext) -> HleResult {
    let p = ctx.rdi();
    if p != 0 {
        crate::font::gfree(p);
    }
    0
}

pub(super) const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("zvzWA5IZMsg", "sceNetResolverStartAton6", cb_resolver_start_aton6),
    ("eyLyLJrdEOU", "sceNetClearDnsCache", cb_clear_dns_cache),
    ("nCL0NyZsd5A", "sceNetGetDnsInfo", cb_get_dns_info),
    ("B-M6KjO8-+w", "sceNetSetDnsInfo", cb_set_dns_info),
    ("15Ywg-ZsSl0", "sceNetSetDns6Info", cb_set_dns6_info),
    ("wmoIm94hqik", "sceNetGetDns6Info", cb_get_dns6_info),
    ("p2vxsE2U3RQ", "sceNetGetSystemTime", cb_get_system_time),
    ("G3O2j9f5z00", "sceNetGetRandom", cb_get_random),
    ("0MT2l3uIX7c", "sceNetGetIfName", cb_get_if_name),
    ("HKIa-WH0AZ4", "sceNetMemoryAllocate", cb_memory_allocate),
    ("221fvqVs+sQ", "sceNetMemoryFree", cb_memory_free),
    ("cEMX1VcPpQ8", "sceNetShowRoute", cb_show),
    ("fCa7-ihdRdc", "sceNetShowRoute6", cb_show),
    ("TCkRD0DWNLg", "sceNetGetpeername", cb_getpeername),
    ("3CHi1K1wsCQ", "sceNetHtonll", cb_htonll),
    ("tOrRi-v3AOM", "sceNetNtohll", cb_htonll),
    ("wvuUDv0jrMI", "sceNetRecvmsg", cb_recvmsg),
    ("2eKbgcboJso", "sceNetSendmsg", cb_sendmsg),
    ("zl35YNs9jnI", "sceNetResolverStartNtoa6", cb_resolver_start_ntoa6),
    ("k1V1djYpk7k", "sceNetShowIfconfig", cb_show),
    ("WxislcDAW5I", "sceNetShowNetstat", cb_show),
    ("H5WHYRfDkR0", "sceNetShowNetstatWithMemory", cb_show_netstat_with_memory),
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CpuRegister::*;

    fn args(ctx: &mut CpuContext, args: &[u64]) {
        for (r, v) in [Rdi, Rsi, Rdx, Rcx, R8, R9].iter().zip(args) {
            ctx.set(*r, *v);
        }
    }

    fn errno(ctx: &CpuContext) -> u32 {
        ctx.read_u32(net_errno_address(ctx)).unwrap()
    }

    /// Whether `cb` fails with SCE_NET_ERROR_<e> and sets errno `e`.
    fn failed(ctx: &mut CpuContext, cb: HleCallback, e: i32) -> bool {
        let r = cb(ctx);
        r == net_error(e) as i64 && errno(ctx) == e as u32
    }

    fn put(ctx: &CpuContext, addr: u64, text: &[u8]) -> u64 {
        let mut b = text.to_vec();
        b.push(0);
        assert!(ctx.write_bytes(addr, &b));
        addr
    }

    /// A sceNet socket bound to 127.0.0.1 on an ephemeral port; returns the
    /// id and its guest address.
    fn bound_socket(ctx: &mut CpuContext, base: u64, ty: u64) -> (u64, [u8; 16]) {
        put(ctx, base, b"test");
        args(ctx, &[base, 2, ty, 0]);
        let s = cb_socket(ctx);
        assert!(s >= SOCKET_FD_MIN as i64, "socket {s:#x}");
        ctx.write_bytes(base + 0x40, &guest_sockaddr_in([127, 0, 0, 1], 0));
        args(ctx, &[s as u64, base + 0x40, 16]);
        assert_eq!(cb_bind(ctx), 0);
        ctx.write_u32(base + 0x60, 16);
        args(ctx, &[s as u64, base + 0x40, base + 0x60]);
        assert_eq!(cb_getsockname(ctx), 0);
        let addr: [u8; 16] = ctx.read_bytes(base + 0x40, 16).unwrap().try_into().unwrap();
        (s as u64, addr)
    }

    fn close(ctx: &mut CpuContext, s: u64) {
        args(ctx, &[s]);
        assert_eq!(cb_socket_close(ctx), 0);
    }

    #[test]
    fn nids_match_names() {
        for &(nid, name, _) in EXPORTS {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    #[test]
    fn byte_order_64() {
        let vm = ps5vm::VirtualMemory::new();
        let mut ctx = CpuContext::new(&vm);
        ctx.set(Rdi, 0x0102_0304_0506_0708);
        assert_eq!(cb_htonll(&mut ctx) as u64, 0x0807_0605_0403_0201);
    }

    /// Scatter/gather over UDP loopback (reference CheckMessages).
    #[test]
    fn messages_scatter_and_gather() {
        let _ids = super::super::tests::socket_ids();
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5E11_0000_0000;
        vm.map(BASE, 0x4000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let (receiver, address) = bound_socket(&mut ctx, BASE, 2);
        let (sender, _) = bound_socket(&mut ctx, BASE + 0x100, 2);

        // send: "scatter " + "" + "gather" to the receiver.
        let (head, tail, dest, out_iov, send) = (BASE + 0x200, BASE + 0x220, BASE + 0x240, BASE + 0x260, BASE + 0x300);
        ctx.write_bytes(head, b"scatter ");
        ctx.write_bytes(tail, b"gather");
        ctx.write_bytes(dest, &address);
        for (i, (b, l)) in [(head, 8u64), (0, 0), (tail, 6)].iter().enumerate() {
            ctx.write_u64(out_iov + i as u64 * 16, *b);
            ctx.write_u64(out_iov + i as u64 * 16 + 8, *l);
        }
        let mut hdr = [0u8; MSGHDR_SIZE];
        hdr[0..8].copy_from_slice(&dest.to_le_bytes());
        hdr[8..12].copy_from_slice(&16u32.to_le_bytes());
        hdr[16..24].copy_from_slice(&out_iov.to_le_bytes());
        hdr[24..28].copy_from_slice(&3u32.to_le_bytes());
        ctx.write_bytes(send, &hdr);
        args(&mut ctx, &[sender, send, 0]);
        assert_eq!(cb_sendmsg(&mut ctx), 14);

        // receive into 5 + 20 bytes, with peek first.
        let (first, second, source, control, in_iov, recv) =
            (BASE + 0x400, BASE + 0x410, BASE + 0x440, BASE + 0x480, BASE + 0x4C0, BASE + 0x500);
        for (i, (b, l)) in [(first, 5u64), (second, 20)].iter().enumerate() {
            ctx.write_u64(in_iov + i as u64 * 16, *b);
            ctx.write_u64(in_iov + i as u64 * 16 + 8, *l);
        }
        let mut hdr = [0u8; MSGHDR_SIZE];
        hdr[0..8].copy_from_slice(&source.to_le_bytes());
        hdr[8..12].copy_from_slice(&28u32.to_le_bytes());
        hdr[16..24].copy_from_slice(&in_iov.to_le_bytes());
        hdr[24..28].copy_from_slice(&2u32.to_le_bytes());
        hdr[32..40].copy_from_slice(&control.to_le_bytes());
        hdr[40..44].copy_from_slice(&16u32.to_le_bytes());
        hdr[44..48].copy_from_slice(&(-1i32).to_le_bytes());
        ctx.write_bytes(recv, &hdr);
        args(&mut ctx, &[receiver, recv, MSG_PEEK as u64]);
        assert_eq!(cb_recvmsg(&mut ctx), 14);
        assert_eq!(ctx.read_bytes(first, 5).unwrap(), b"scatt");
        assert_eq!(ctx.read_bytes(second, 9).unwrap(), b"er gather");
        assert_eq!(ctx.read_u32(recv + MSGHDR_FLAGS), Some(0));
        assert_eq!(ctx.read_u32(recv + MSGHDR_CONTROL_LENGTH), Some(0));
        assert_eq!(ctx.read_u32(recv + MSGHDR_NAME_LENGTH), Some(16));
        let src = ctx.read_bytes(source, 8).unwrap();
        assert_eq!((src[0], src[1], src[4], src[7]), (16, 2, 127, 1));
        // One 5-byte buffer: the datagram is cut and MSG_TRUNC reported.
        ctx.write_bytes(second, &[0; 20]);
        ctx.write_u32(recv + 24, 1);
        ctx.write_u32(recv + MSGHDR_FLAGS, u32::MAX);
        args(&mut ctx, &[receiver, recv, 0]);
        assert_eq!(cb_recvmsg(&mut ctx), 5);
        assert_eq!(ctx.read_u32(recv + MSGHDR_FLAGS), Some(MSG_TRUNC));
        assert_eq!(ctx.read_bytes(second, 1).unwrap(), [0]);

        // Unwritable outputs fail before the receive: the datagram stays.
        // Never mapped (the first pages of the address space).
        let unmapped = 0x1000;
        args(&mut ctx, &[sender, send, 0]);
        assert_eq!(cb_sendmsg(&mut ctx), 14);
        ctx.write_u64(in_iov, unmapped);
        args(&mut ctx, &[receiver, recv, 0]);
        assert!(failed(&mut ctx, cb_recvmsg, EFAULT_));
        args(&mut ctx, &[receiver, unmapped, 32, 0, source, recv + 8]);
        ctx.write_u32(recv + 8, 16);
        assert!(failed(&mut ctx, super::super::cb_recvfrom, EFAULT_));
        args(&mut ctx, &[receiver, first, 32, 0, unmapped, recv + 8]);
        ctx.write_u32(recv + 8, 16);
        assert!(failed(&mut ctx, super::super::cb_recvfrom, EFAULT_));
        args(&mut ctx, &[receiver, first, 32, 0, source, recv + 8]);
        ctx.write_u32(recv + 8, 16);
        assert_eq!(super::super::cb_recvfrom(&mut ctx), 14);
        assert_eq!(ctx.read_bytes(first, 14).unwrap(), b"scatter gather");
        ctx.write_u64(in_iov, first);

        // Header validation.
        args(&mut ctx, &[sender, 0, 0]);
        assert!(failed(&mut ctx, cb_sendmsg, EFAULT_));
        args(&mut ctx, &[receiver, 0, 0]);
        assert!(failed(&mut ctx, cb_recvmsg, EFAULT_));
        let bad = BASE + 0x600;
        let set_bad = |ctx: &CpuContext, iov: u64, len: i32| {
            let mut h = [0u8; MSGHDR_SIZE];
            h[16..24].copy_from_slice(&iov.to_le_bytes());
            h[24..28].copy_from_slice(&len.to_le_bytes());
            ctx.write_bytes(bad, &h);
        };
        set_bad(&ctx, out_iov, -1);
        args(&mut ctx, &[sender, bad, 0]);
        assert!(failed(&mut ctx, cb_sendmsg, EMSGSIZE_));
        set_bad(&ctx, out_iov, 1025);
        args(&mut ctx, &[receiver, bad, 0]);
        assert!(failed(&mut ctx, cb_recvmsg, EMSGSIZE_));
        set_bad(&ctx, 0, 1);
        args(&mut ctx, &[sender, bad, 0]);
        assert!(failed(&mut ctx, cb_sendmsg, EFAULT_));
        let hole = BASE + 0x700;
        ctx.write_u64(hole, 0);
        ctx.write_u64(hole + 8, 1);
        set_bad(&ctx, hole, 1);
        args(&mut ctx, &[receiver, bad, 0]);
        assert!(failed(&mut ctx, cb_recvmsg, EFAULT_));
        ctx.write_u64(hole, head);
        ctx.write_u64(hole + 8, 0x7FFF_FFFF);
        ctx.write_u64(hole + 16, tail);
        ctx.write_u64(hole + 24, 1);
        set_bad(&ctx, hole, 2);
        args(&mut ctx, &[sender, bad, 0]);
        assert!(failed(&mut ctx, cb_sendmsg, EINVAL_));
        args(&mut ctx, &[receiver, recv, 1]);
        assert!(failed(&mut ctx, cb_recvmsg, EOPNOTSUPP_));
        args(&mut ctx, &[sender, send, 1]);
        assert!(failed(&mut ctx, cb_sendmsg, EOPNOTSUPP_));
        args(&mut ctx, &[12345, recv, 0]);
        assert!(failed(&mut ctx, cb_recvmsg, EBADF_));
        args(&mut ctx, &[12345, send, 0]);
        assert!(failed(&mut ctx, cb_sendmsg, EBADF_));
        // Control data cannot be sent.
        ctx.write_u64(send + 32, control);
        ctx.write_u32(send + 40, 16);
        args(&mut ctx, &[sender, send, 0]);
        assert!(failed(&mut ctx, cb_sendmsg, EOPNOTSUPP_));

        close(&mut ctx, receiver);
        close(&mut ctx, sender);
    }

    /// The peer name copies as much as the capacity holds (reference
    /// accepted-socket case).
    #[test]
    fn peer_names() {
        let _ids = super::super::tests::socket_ids();
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5E11_0010_0000;
        vm.map(BASE, 0x4000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let (listener, address) = bound_socket(&mut ctx, BASE, 1);
        args(&mut ctx, &[listener, 1]);
        assert_eq!(cb_listen(&mut ctx), 0);
        put(&ctx, BASE + 0x100, b"client");
        args(&mut ctx, &[BASE + 0x100, 2, 1, 0]);
        let client = cb_socket(&mut ctx) as u64;
        ctx.write_bytes(BASE + 0x140, &address);
        args(&mut ctx, &[client, BASE + 0x140, 16]);
        assert_eq!(cb_connect(&mut ctx), 0);
        ctx.write_u32(BASE + 0x160, 16);
        args(&mut ctx, &[client, BASE + 0x170, BASE + 0x160]);
        assert_eq!(cb_getsockname(&mut ctx), 0);
        let client_address = ctx.read_bytes(BASE + 0x170, 16).unwrap();
        args(&mut ctx, &[listener, 0, 0]);
        let accepted = cb_accept(&mut ctx);
        assert!(accepted >= 0);
        let accepted = accepted as u64;
        for capacity in [0u32, 1, 15, 16, 28, u32::MAX] {
            let copied = capacity.min(16) as usize;
            ctx.write_bytes(BASE + 0x200, &[0xA5; 28]);
            ctx.write_u32(BASE + 0x240, capacity);
            args(&mut ctx, &[accepted, BASE + 0x200, BASE + 0x240]);
            assert_eq!(cb_getpeername(&mut ctx), 0);
            assert_eq!(ctx.read_u32(BASE + 0x240), Some(copied as u32));
            let peer = ctx.read_bytes(BASE + 0x200, 28).unwrap();
            assert_eq!(&peer[..copied], &client_address[..copied]);
            assert!(peer[copied..].iter().all(|&b| b == 0xA5));
        }
        args(&mut ctx, &[accepted, 0, BASE + 0x240]);
        assert!(failed(&mut ctx, cb_getpeername, EINVAL_));
        args(&mut ctx, &[12345, BASE + 0x200, BASE + 0x240]);
        assert!(failed(&mut ctx, cb_getpeername, EBADF_));
        // A listening socket has no peer.
        args(&mut ctx, &[listener, BASE + 0x200, BASE + 0x240]);
        assert!(failed(&mut ctx, cb_getpeername, ENOTCONN_));
        for s in [accepted, client, listener] {
            close(&mut ctx, s);
        }
    }

    /// A socket accepted from a non-blocking listener is non-blocking too
    /// (FreeBSD accept inherits the mode); a blocking listener's is blocking.
    #[test]
    fn accepted_socket_inherits_non_blocking_mode() {
        let _ids = super::super::tests::socket_ids();
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5E11_0050_0000;
        vm.map(BASE, 0x4000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let (listener, address) = bound_socket(&mut ctx, BASE, 1);
        args(&mut ctx, &[listener, 4]);
        assert_eq!(cb_listen(&mut ctx), 0);
        let nbio = |ctx: &mut CpuContext, s: u64| {
            ctx.write_u32(BASE + 0x310, 4);
            args(ctx, &[s, SOL_SOCKET as u64, SO_NBIO as u64, BASE + 0x300, BASE + 0x310]);
            assert_eq!(cb_getsockopt(ctx), 0);
            ctx.read_u32(BASE + 0x300).unwrap()
        };
        let mut accepted = Vec::new();
        for (round, on) in [(0u64, 1u32), (1, 0)] {
            ctx.write_u32(BASE + 0x320, on);
            args(&mut ctx, &[listener, SOL_SOCKET as u64, SO_NBIO as u64, BASE + 0x320, 4]);
            assert_eq!(cb_setsockopt(&mut ctx), 0);
            put(&ctx, BASE + 0x100, b"client");
            args(&mut ctx, &[BASE + 0x100, 2, 1, 0]);
            let client = cb_socket(&mut ctx) as u64;
            ctx.write_bytes(BASE + 0x140, &address);
            args(&mut ctx, &[client, BASE + 0x140, 16]);
            assert_eq!(cb_connect(&mut ctx), 0);
            let mut a = -1;
            for _ in 0..200 {
                args(&mut ctx, &[listener, 0, 0]);
                a = cb_accept(&mut ctx);
                if a >= 0 {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            assert!(a >= 0, "round {round}");
            assert_eq!(nbio(&mut ctx, a as u64), on, "round {round}");
            if on != 0 {
                args(&mut ctx, &[a as u64, BASE + 0x400, 8, 0]);
                assert!(failed(&mut ctx, super::super::cb_recv, EWOULDBLOCK_));
            }
            accepted.push(a as u64);
            accepted.push(client);
        }
        for s in accepted.into_iter().chain([listener]) {
            close(&mut ctx, s);
        }
    }

    /// sceNetSetsockopt and sceNetSocketClose take a libkernel socket()
    /// descriptor too.
    #[test]
    fn kernel_socket_through_scenet() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5E11_0060_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let fd = super::super::kernel_socket_core(2, 1, 0).unwrap();
        ctx.write_u32(BASE, 1);
        args(&mut ctx, &[fd as u64, 6, 1, BASE, 4]);
        assert_eq!(cb_setsockopt(&mut ctx), 0, "TCP_NODELAY");
        ctx.write_u32(BASE, 65536);
        args(&mut ctx, &[fd as u64, SOL_SOCKET as u64, 0x1001, BASE, 4]);
        assert_eq!(cb_setsockopt(&mut ctx), 0, "SO_SNDBUF");
        args(&mut ctx, &[fd as u64]);
        assert_eq!(cb_socket_close(&mut ctx), 0);
        assert!(!super::super::is_kernel_socket(fd));
        args(&mut ctx, &[0x7fff_0000]);
        assert!(failed(&mut ctx, cb_socket_close, EBADF_));
        args(&mut ctx, &[0x7fff_0000, 6, 1, BASE, 4]);
        assert!(failed(&mut ctx, cb_setsockopt, EBADF_));
    }

    #[test]
    fn resolver_ntoa6_offline() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5E11_0020_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        put(&ctx, BASE, b"pool");
        let pool = tests_pool(&mut ctx, BASE);
        put(&ctx, BASE + 0x20, b"resolver");
        args(&mut ctx, &[BASE + 0x20, pool, 0]);
        let rid = cb_resolver_create(&mut ctx) as u64;
        let (out, status) = (BASE + 0x100, BASE + 0x200);
        ctx.write_bytes(out, &[0xA5; 16]);
        put(&ctx, BASE + 0x40, b"::1");
        args(&mut ctx, &[rid, BASE + 0x40, out, 5_000_000, 1, 0]);
        assert_eq!(cb_resolver_start_ntoa6(&mut ctx), 0);
        assert_eq!(ctx.read_bytes(out, 16).unwrap(), std::net::Ipv6Addr::LOCALHOST.octets());
        args(&mut ctx, &[rid, status]);
        assert_eq!(cb_resolver_get_error(&mut ctx), 0);
        assert_eq!(ctx.read_u32(status), Some(0));
        args(&mut ctx, &[rid, 0, out, 5_000_000, 1, 0]);
        assert!(failed(&mut ctx, cb_resolver_start_ntoa6, EINVAL_));
        for name in [&b"127.0.0.1"[..], b"example.com"] {
            put(&ctx, BASE + 0x40, name);
            args(&mut ctx, &[rid, BASE + 0x40, out, 5_000_000, 1, 0]);
            assert_eq!(cb_resolver_start_ntoa6(&mut ctx), NET_ERROR_RESOLVER_NO_DNS as i64);
            args(&mut ctx, &[rid, status]);
            assert_eq!(cb_resolver_get_error(&mut ctx), 0);
            assert_eq!(ctx.read_u32(status), Some(NET_ERROR_RESOLVER_NO_DNS as u32));
        }
        args(&mut ctx, &[rid, BASE + 0x40, out, 5_000_000, 1, 1]);
        assert!(failed(&mut ctx, cb_resolver_start_ntoa6, EINVAL_));
        args(&mut ctx, &[31, BASE + 0x40, out, 5_000_000, 1, 0]);
        if net().resolver_mut(31).is_none() {
            assert!(failed(&mut ctx, cb_resolver_start_ntoa6, EBADF_));
        }
        args(&mut ctx, &[rid]);
        assert_eq!(cb_resolver_destroy(&mut ctx), 0);
    }

    /// Reference CheckNetstat: the dump lists the pool and its sockets.
    #[test]
    fn netstat_dump() {
        let _ids = super::super::tests::socket_ids();
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5E11_0030_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        for cb in [cb_show as HleCallback, cb_show] {
            assert_eq!(cb(&mut ctx), 0);
        }
        args(&mut ctx, &[-1i64 as u64]);
        assert!(failed(&mut ctx, cb_show_netstat_with_memory, EBADF_));
        put(&ctx, BASE + 0x200, b"netstat");
        args(&mut ctx, &[BASE + 0x200, 4096, 0]);
        let pool = cb_pool_create(&mut ctx);
        assert!(pool >= 0);
        let (s, _) = bound_socket(&mut ctx, BASE, 1);
        let nonblocking = BASE + 0x300;
        ctx.write_u32(nonblocking, 1);
        args(&mut ctx, &[s, 0xFFFF, 0x1200, nonblocking, 4]);
        assert_eq!(cb_setsockopt(&mut ctx), 0);
        args(&mut ctx, &[s, 1]);
        assert_eq!(cb_listen(&mut ctx), 0);
        let lines = netstat_lines(pool as i32).unwrap();
        assert!(lines[0].contains("bytes=4096 sockets="), "{lines:?}");
        assert!(lines.contains(&format!("socket={s} family=2 type=1 nonblock=1 bound=1 listening=1")), "{lines:?}");
        args(&mut ctx, &[pool as u64]);
        assert_eq!(cb_show_netstat_with_memory(&mut ctx), 0);
        close(&mut ctx, s);
        args(&mut ctx, &[pool as u64]);
        assert_eq!(cb_pool_destroy(&mut ctx), 0);
        assert!(failed(&mut ctx, cb_show_netstat_with_memory, EBADF_));
    }

    /// sceNetSetDnsInfo / SetDns6Info replace the servers GetDnsInfo and
    /// GetDns6Info report (count 0, 1 or 2); flags or a zero primary are
    /// EINVAL and change nothing; NULL goes back to none.
    #[test]
    fn dns_servers_set_and_read_back() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5E11_0070_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let (info, read) = (BASE, BASE + 0x100);
        args(&mut ctx, &[read, 0]);
        assert_eq!(cb_get_dns_info(&mut ctx), 0);
        ctx.write_u32(info, 0x0808_0808);
        ctx.write_u32(info + 4, 0);
        args(&mut ctx, &[info, 1]);
        assert!(failed(&mut ctx, cb_set_dns_info, EINVAL_));
        args(&mut ctx, &[info, 0]);
        assert_eq!(cb_set_dns_info(&mut ctx), 0);
        args(&mut ctx, &[read, 0]);
        assert_eq!(cb_get_dns_info(&mut ctx), 1);
        assert_eq!(ctx.read_u32(read), Some(0x0808_0808));
        ctx.write_u32(info + 4, 0x0101_0101);
        args(&mut ctx, &[info, 0]);
        assert_eq!(cb_set_dns_info(&mut ctx), 0);
        args(&mut ctx, &[read, 0]);
        assert_eq!(cb_get_dns_info(&mut ctx), 2);
        assert_eq!(ctx.read_u32(read + 4), Some(0x0101_0101));
        ctx.write_u32(info, 0);
        args(&mut ctx, &[info, 0]);
        assert!(failed(&mut ctx, cb_set_dns_info, EINVAL_));
        args(&mut ctx, &[read, 0]);
        assert_eq!(cb_get_dns_info(&mut ctx), 2, "a refused set keeps the servers");
        args(&mut ctx, &[0, 0]);
        assert_eq!(cb_set_dns_info(&mut ctx), 0);
        args(&mut ctx, &[read, 0]);
        assert_eq!(cb_get_dns_info(&mut ctx), 0);
        assert_eq!(ctx.read_u64(read), Some(0));

        args(&mut ctx, &[read, 0]);
        assert_eq!(cb_get_dns6_info(&mut ctx), 0);
        args(&mut ctx, &[0, 0]);
        assert!(failed(&mut ctx, cb_get_dns6_info, EINVAL_));
        ctx.write_bytes(info, &[0; 32]);
        args(&mut ctx, &[info, 0]);
        assert!(failed(&mut ctx, cb_set_dns6_info, EINVAL_));
        ctx.write_bytes(info, &[0x20; 16]);
        assert_eq!(cb_set_dns6_info(&mut ctx), 0);
        args(&mut ctx, &[read, 0]);
        assert_eq!(cb_get_dns6_info(&mut ctx), 1);
        assert_eq!(ctx.read_bytes(read, 16).unwrap(), vec![0x20; 16]);
        args(&mut ctx, &[0, 0]);
        assert_eq!(cb_set_dns6_info(&mut ctx), 0);
        args(&mut ctx, &[read, 0]);
        assert_eq!(cb_get_dns6_info(&mut ctx), 0);
    }
}
