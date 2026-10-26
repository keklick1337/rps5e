// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceSsl — contexts with their certificate store, certificate readers
//! and TLS client connections over guest sockets.
//!
//! A context keeps the CA certificates and the client identity loaded with
//! sceSslLoadCert. No root bundle is shipped, so an empty context has no CA
//! and sceSslConnect refuses to run without one. Loaded certificates and CA
//! snapshots are charged to the context's pool (DER bytes and snapshot
//! arrays); TLS session state is the host library's and is not charged.
//!
//! sceSslConnect runs a TLS 1.2 client handshake over the host socket of a
//! libkernel or libSceNet socket: the server's chain, validity dates and
//! host name (also sent as SNI) are checked against the loaded CAs, and the
//! loaded client certificate is offered. Where the console's error code for
//! a failure is unknown, the closest SCE_SSL_ERROR code is returned and the
//! failure is logged.

#[cfg(windows)]
#[allow(unused_imports)]
use crate::host::c as libc;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard, Weak};

use openssl::hash::MessageDigest;
use openssl::pkey::{PKey, Private};
use openssl::ssl::{ErrorCode, SslConnector, SslMethod, SslOptions, SslStream, SslVerifyMode, SslVersion};
use openssl::x509::store::X509StoreBuilder;
use openssl::x509::{X509NameRef, X509VerifyResult, X509};

use crate::context::{CpuContext, CpuRegister};
use crate::registry::{HleCallback, HleRegistry, HleResult};

const LIB: &str = "libSceSsl";

const fn err(v: u32) -> HleResult {
    v as i32 as i64
}

const SSL_ERROR_BROKEN: HleResult = err(0x8095_F003);
const SSL_ERROR_NOT_FOUND: HleResult = err(0x8095_F004);
const SSL_ERROR_INVALID_FORMAT: HleResult = err(0x8095_F005);
const SSL_ERROR_INVALID_ID: HleResult = err(0x8095_F006);
const SSL_ERROR_INVALID_VALUE: HleResult = err(0x8095_F007);
const SSL_ERROR_OUT_OF_SIZE: HleResult = err(0x8095_F008);
const SSL_ERROR_INVALID_CERT: HleResult = err(0x8095_F00A);
const SSL_ERROR_CN_CHECK: HleResult = err(0x8095_F00B);
const SSL_ERROR_UNKNOWN_CA: HleResult = err(0x8095_F00C);
const SSL_ERROR_NOT_AFTER_CHECK: HleResult = err(0x8095_F00D);
const SSL_ERROR_NOT_BEFORE_CHECK: HleResult = err(0x8095_F00E);
const SSL_ERROR_FATAL_ALERT: HleResult = err(0x8095_F010);
const SSL_ERROR_BUSY: HleResult = err(0x8095_F012);
const SSL_ERROR_WANT_POLLIN: HleResult = err(0x8095_F013);
const SSL_ERROR_WANT_POLLOUT: HleResult = err(0x8095_F014);
const SSL_ERROR_SOCKET_CLOSED: HleResult = err(0x8095_170D);
const SSL_ERROR_INVALID_ARG: HleResult = err(0x8095_177A);
const SSL_ERROR_OUT_OF_MEMORY: HleResult = err(0x8095_17D5);
const SSL_ERROR_MISMATCH_PUBLIC_KEYS: HleResult = err(0x8095_20D3);
const ERR_MEMORY_FAULT: HleResult = err(0x8002_000E);

/// SceSslData: ptr, size.
const DATA_SIZE: u64 = 16;
/// SceSslCaCerts: certData*, certDataNum (size_t), pool*.
const CA_CERTS_SIZE: usize = 24;
/// SceSslCaList: certs (void**), num (int), padding.
const CA_LIST_SIZE: usize = 16;
/// Opaque guest object standing for a certificate or a name.
const HANDLE_SIZE: u64 = 16;

/// Context ids carry the object type in their high half (2 = SSL) and the
/// slot index in the low half.
const SSL_ID_TYPE: i32 = 2 << 16;
const SSL_MAX: usize = 32;

/// sceSslRead flag: return what one read delivers.
const READ_PARTIAL: i32 = 0x1;
/// Highest SceSslVersion (TLS 1.2).
const SSL_VERSION_MAX: i32 = 5;
/// The verify options a connection starts with (measured on a PS5, 13.40):
/// SERVER_VERIFY, CN_CHECK and KNOWN_CA_CHECK; the date checks are off.
const DEFAULT_VERIFY_OPTIONS: u32 = 0x25;
/// The verify option bits sceSslEnable/DisableVerifyOption take.
const VERIFY_OPTIONS_MASK: u32 = 0x3F;
/// Longest host name read for a connection.
const MAX_HOST: usize = 1024;
/// Largest single sceSslRead / sceSslWrite transfer handled at once.
const MAX_TRANSFER: u64 = 4 << 20;

// ---------------------------------------------------------------------------
// Pool accounting
// ---------------------------------------------------------------------------

/// A context's memory pool: size, bytes in use and the peak.
struct Pool {
    size: u64,
    usage: Mutex<(u64, u64)>,
}

impl Pool {
    fn new(size: u64) -> Arc<Self> {
        Arc::new(Self { size, usage: Mutex::new((0, 0)) })
    }

    fn usage(&self) -> MutexGuard<'_, (u64, u64)> {
        self.usage.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Take `bytes` from the pool, released when the charge drops; None
    /// when the pool cannot hold them.
    fn charge(self: &Arc<Self>, bytes: u64) -> Option<Charge> {
        let mut u = self.usage();
        if bytes > self.size - u.0 {
            return None;
        }
        u.0 += bytes;
        u.1 = u.1.max(u.0);
        Some(Charge { pool: self.clone(), bytes })
    }
}

/// Pool bytes held by an object.
struct Charge {
    pool: Arc<Pool>,
    bytes: u64,
}

impl Drop for Charge {
    fn drop(&mut self) {
        self.pool.usage().0 -= self.bytes;
    }
}

fn charge(pool: &Arc<Pool>, bytes: u64, what: &str) -> Result<Charge, HleResult> {
    pool.charge(bytes).ok_or_else(|| {
        log::warn!("[ssl] {what}: the context pool ({} bytes) cannot hold {bytes} more bytes", pool.size);
        SSL_ERROR_OUT_OF_MEMORY
    })
}

// ---------------------------------------------------------------------------
// Certificates
// ---------------------------------------------------------------------------

/// Certificates loaded from one SceSslData, with their guest handles.
struct CertChain {
    certs: Vec<X509>,
    handles: Vec<u64>,
    _charge: Charge,
}

/// A client certificate with its private key.
struct Identity {
    chain: Arc<CertChain>,
    key: PKey<Private>,
    _charge: Charge,
}

/// A CA list or CA certificate array handed to the guest: its guest block
/// and the chains it refers to stay alive until it is freed.
struct Snapshot {
    block: u64,
    count: usize,
    _chains: Vec<Arc<CertChain>>,
    _charge: Charge,
}

impl Drop for Snapshot {
    fn drop(&mut self) {
        crate::font::gfree(self.block);
    }
}

/// SceSslCertName: the (OID name, value) pairs of a subject or issuer.
struct CertName {
    entries: Vec<(String, Vec<u8>)>,
}

struct SslContext {
    pool: Arc<Pool>,
    authorities: Vec<Arc<CertChain>>,
    identity: Option<Arc<Identity>>,
    /// Guest certificate handle -> its certificate while a chain holding
    /// it is alive.
    handles: HashMap<u64, (Weak<CertChain>, usize)>,
    ca_lists: HashMap<u64, Snapshot>,
    ca_certs: HashMap<u64, Snapshot>,
    names: HashMap<u64, CertName>,
}

impl SslContext {
    fn new(pool_size: u64) -> Self {
        Self {
            pool: Pool::new(pool_size),
            authorities: Vec::new(),
            identity: None,
            handles: HashMap::new(),
            ca_lists: HashMap::new(),
            ca_certs: HashMap::new(),
            names: HashMap::new(),
        }
    }

    fn certificate_count(&self) -> usize {
        self.authorities.iter().map(|c| c.handles.len()).sum()
    }

    /// The certificate of a guest handle: INVALID_VALUE for NULL,
    /// INVALID_CERT for a handle not handed out or whose chain is gone.
    fn certificate(&self, handle: u64, function: &str) -> Result<X509, HleResult> {
        if handle == 0 {
            return Err(SSL_ERROR_INVALID_VALUE);
        }
        let Some((chain, index)) = self.handles.get(&handle) else {
            log::warn!("[ssl] {function}: {handle:#x} is not a certificate of the context");
            return Err(SSL_ERROR_INVALID_CERT);
        };
        match chain.upgrade() {
            Some(chain) => Ok(chain.certs[*index].clone()),
            None => {
                log::warn!("[ssl] {function}: certificate {handle:#x} was unloaded");
                Err(SSL_ERROR_INVALID_CERT)
            }
        }
    }
}

impl Drop for SslContext {
    fn drop(&mut self) {
        for (&handle, _) in self.handles.iter() {
            crate::font::gfree(handle);
        }
        for (&name, _) in self.names.iter() {
            crate::font::gfree(name);
        }
    }
}

// ---------------------------------------------------------------------------
// Connections
// ---------------------------------------------------------------------------

/// A guest socket's host descriptor, kept open while the connection uses it.
enum HostSocket {
    Kernel(crate::net::PollHandle),
    Net(crate::net::NetPollHandle),
}

impl HostSocket {
    /// The host socket of a libkernel descriptor or libSceNet socket id.
    fn of(id: i32) -> Option<Self> {
        if let Some(h) = crate::net::kernel_socket_poll_handle(id) {
            return Some(Self::Kernel(h));
        }
        match crate::net::net_socket_poll_handle(id) {
            Some(Ok(h)) => Some(Self::Net(h)),
            _ => None,
        }
    }

    fn fd(&self) -> i32 {
        match self {
            Self::Kernel(h) => h.fd(),
            Self::Net(h) => h.fd(),
        }
    }
}

impl Read for HostSocket {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        loop {
            let r = unsafe { libc::recv(self.fd(), buf.as_mut_ptr() as *mut libc::c_void, buf.len(), 0) };
            if r >= 0 {
                return Ok(r as usize);
            }
            let e = std::io::Error::last_os_error();
            if e.kind() != std::io::ErrorKind::Interrupted {
                return Err(e);
            }
        }
    }
}

impl Write for HostSocket {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        loop {
            let r = unsafe { libc::send(self.fd(), buf.as_ptr() as *const libc::c_void, buf.len(), libc::MSG_NOSIGNAL) };
            if r >= 0 {
                return Ok(r as usize);
            }
            let e = std::io::Error::last_os_error();
            if e.kind() != std::io::ErrorKind::Interrupted {
                return Err(e);
            }
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

type Session = Arc<Mutex<SslStream<HostSocket>>>;

struct SslConnection {
    context: i32,
    socket: i32,
    host: String,
    /// sceSslSetVerifyCallback: how the console calls it is unknown, so a
    /// connection with one refuses to connect.
    verify_callback: u64,
    #[allow(dead_code)]
    verify_arg: u64,
    #[allow(dead_code)]
    verify_options: u32,
    #[allow(dead_code)]
    min_version: i32,
    /// sceSslSetAlpn: kept, not offered (no protocol is ever selected).
    #[allow(dead_code)]
    alpn: Vec<u8>,
    session: Option<Session>,
}

struct SslState {
    contexts: Vec<Option<SslContext>>,
    connections: HashMap<i32, SslConnection>,
    next_connection: i32,
    /// Slot after the last context created: slots are handed out in turn,
    /// so a terminated context's id is not reused at once.
    next_slot: usize,
}

static SSL: LazyLock<Mutex<SslState>> =
    LazyLock::new(|| Mutex::new(SslState { contexts: (0..SSL_MAX).map(|_| None).collect(), connections: HashMap::new(), next_connection: 1, next_slot: 0 }));

fn ssl() -> MutexGuard<'static, SslState> {
    SSL.lock().unwrap_or_else(|p| p.into_inner())
}

/// Slot of a context id, if it has the SSL type.
fn slot_of(id: i32) -> Option<usize> {
    if id & !0xFFFF != SSL_ID_TYPE {
        return None;
    }
    let slot = (id & 0xFFFF) as usize;
    (slot < SSL_MAX).then_some(slot)
}

impl SslState {
    fn context(&mut self, id: i32) -> Option<&mut SslContext> {
        slot_of(id).and_then(|s| self.contexts[s].as_mut())
    }

    /// A connection whose context is alive; INVALID_ID (logged) otherwise.
    fn connection(&mut self, id: i32, function: &str) -> Result<&mut SslConnection, HleResult> {
        let alive = match self.connections.get(&id) {
            None => {
                log::warn!("[ssl] {function}: unknown connection {id:#x}");
                return Err(SSL_ERROR_INVALID_ID);
            }
            Some(c) => slot_of(c.context).is_some_and(|s| self.contexts[s].is_some()),
        };
        if !alive {
            log::warn!("[ssl] {function}: the context of connection {id:#x} was terminated");
            return Err(SSL_ERROR_INVALID_ID);
        }
        Ok(self.connections.get_mut(&id).unwrap())
    }
}

/// Whether `id` is a live sceSslInit context.
pub(crate) fn ssl_valid(id: i32) -> bool {
    ssl().context(id).is_some()
}

fn stack_arg(ctx: &CpuContext, index: u64) -> u64 {
    ctx.read_u64(ctx.get(CpuRegister::Rsp) + 8 + 8 * index).unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Contexts
// ---------------------------------------------------------------------------

/// sceSslInit(poolSize) -> context id; a pool size of 0 is the kernel's
/// EINVAL (measured on a PS5, 13.40).
fn cb_ssl_init(ctx: &mut CpuContext) -> HleResult {
    const KERNEL_EINVAL: HleResult = err(0x8002_0016);
    let pool_size = ctx.rdi();
    if pool_size == 0 {
        return KERNEL_EINVAL;
    }
    let mut st = ssl();
    let start = st.next_slot;
    match (0..SSL_MAX).map(|i| (start + i) % SSL_MAX).find(|&s| st.contexts[s].is_none()) {
        Some(slot) => {
            st.contexts[slot] = Some(SslContext::new(pool_size));
            st.next_slot = (slot + 1) % SSL_MAX;
            (SSL_ID_TYPE | slot as i32) as i64
        }
        None => SSL_ERROR_OUT_OF_SIZE,
    }
}

/// sceSslTerm(ctxId): the context's certificates, snapshots and names go,
/// and so do its connections (sceSslDeleteConnection then answers
/// INVALID_ID), so a later context in the same slot cannot revive them.
fn cb_ssl_term(ctx: &mut CpuContext) -> HleResult {
    let id = ctx.rdi() as i32;
    let mut st = ssl();
    let dead = slot_of(id).and_then(|s| st.contexts[s].take());
    match dead {
        Some(_) => {
            st.connections.retain(|_, c| c.context != id);
            0
        }
        None => SSL_ERROR_INVALID_ID,
    }
}

/// Test hook: sceSslInit with the pool size in rdi.
#[cfg(test)]
pub(crate) fn tests_init(ctx: &mut CpuContext) -> u64 {
    cb_ssl_init(ctx) as u64
}

/// sceSslClose(connId): an unknown connection is INVALID_ID.
fn cb_ssl_close(ctx: &mut CpuContext) -> HleResult {
    match ssl().connection(ctx.rdi() as i32, "sceSslClose") {
        Ok(_) => 0,
        Err(e) => e,
    }
}

/// sceSslGetMemoryPoolStats(ctxId, SceSslMemoryPoolStats* stats):
/// poolSize, maxInuseSize, currentInuseSize, reserved.
fn cb_get_memory_pool_stats(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rsi();
    if out == 0 {
        return SSL_ERROR_INVALID_VALUE;
    }
    let pool = ssl().context(ctx.rdi() as i32).map(|c| c.pool.clone());
    let Some(pool) = pool else {
        // An unknown or terminated context zeroes the statistics.
        ctx.write_bytes(out, &[0u8; 32]);
        return SSL_ERROR_INVALID_ID;
    };
    let (current, peak) = *pool.usage();
    let mut b = [0u8; 32];
    b[0..8].copy_from_slice(&pool.size.to_le_bytes());
    b[8..16].copy_from_slice(&peak.to_le_bytes());
    b[16..24].copy_from_slice(&current.to_le_bytes());
    if ctx.write_bytes(out, &b) { 0 } else { ERR_MEMORY_FAULT }
}

// ---------------------------------------------------------------------------
// sceSslLoadCert / sceSslUnloadCert
// ---------------------------------------------------------------------------

/// Certificate or key bytes: PEM (first non-blank byte '-', no embedded
/// NUL) or one DER SEQUENCE with nothing after it.
enum Material {
    Pem(Vec<u8>),
    Der(Vec<u8>),
}

/// Length of the DER SEQUENCE at the start of `b`, header included.
fn der_sequence_length(b: &[u8]) -> Option<usize> {
    if b.len() < 2 || b[0] != 0x30 {
        return None;
    }
    let (len, header) = match b[1] {
        n if n < 0x80 => (n as usize, 2),
        0x81..=0x84 => {
            let count = (b[1] & 0x7F) as usize;
            let bytes = b.get(2..2 + count)?;
            (bytes.iter().fold(0usize, |a, &x| (a << 8) | x as usize), 2 + count)
        }
        _ => return None,
    };
    header.checked_add(len)
}

fn material(bytes: Vec<u8>) -> Result<Material, &'static str> {
    if bytes.is_empty() {
        return Err("empty certificate or key data");
    }
    let first = bytes.iter().position(|c| !matches!(c, b' ' | b'\r' | b'\n' | b'\t'));
    if first.is_some_and(|i| bytes[i] == b'-') {
        let body = bytes.strip_suffix(&[0]).unwrap_or(&bytes);
        if body.contains(&0) {
            return Err("embedded NUL in PEM data");
        }
        return Ok(Material::Pem(body.to_vec()));
    }
    if der_sequence_length(&bytes) != Some(bytes.len()) {
        return Err("not a single DER sequence");
    }
    Ok(Material::Der(bytes))
}

/// Read a guest SceSslData.
fn read_data(ctx: &CpuContext, addr: u64) -> Result<Vec<u8>, HleResult> {
    let ptr = ctx.read_u64(addr).ok_or(ERR_MEMORY_FAULT)?;
    let size = ctx.read_u64(addr + 8).ok_or(ERR_MEMORY_FAULT)?;
    if ptr == 0 || size == 0 {
        log::warn!("[ssl] sceSslLoadCert: empty certificate or key data");
        return Err(SSL_ERROR_INVALID_VALUE);
    }
    if size > MAX_TRANSFER {
        log::warn!("[ssl] sceSslLoadCert: {size} bytes of certificate data");
        return Err(SSL_ERROR_INVALID_VALUE);
    }
    ctx.read_bytes(ptr, size as usize).ok_or(ERR_MEMORY_FAULT)
}

/// Parse the certificates of one SceSslData.
pub(crate) fn parse_certificates(bytes: Vec<u8>) -> Result<Vec<X509>, &'static str> {
    let certs = match material(bytes)? {
        Material::Pem(p) => X509::stack_from_pem(&p).map_err(|_| "invalid PEM certificate chain")?,
        Material::Der(d) => vec![X509::from_der(&d).map_err(|_| "invalid DER certificate")?],
    };
    if certs.is_empty() {
        return Err("no certificate in the data");
    }
    Ok(certs)
}

fn parse_key(bytes: Vec<u8>) -> Result<PKey<Private>, &'static str> {
    match material(bytes)? {
        Material::Pem(p) => PKey::private_key_from_pem(&p),
        Material::Der(d) => PKey::private_key_from_der(&d),
    }
    .map_err(|_| "invalid private key")
}

/// DER size of a certificate list, the bytes a loaded chain is charged.
fn der_size(certs: &[X509]) -> u64 {
    certs.iter().map(|c| c.to_der().map_or(0, |d| d.len() as u64)).sum()
}

fn load_failure(what: &str, code: HleResult) -> HleResult {
    log::warn!("[ssl] sceSslLoadCert: {what}");
    code
}

/// sceSslLoadCert(ctxId, caCertNum, SceSslData** caList, SceSslData* cert,
/// SceSslData* privateKey): the CA chains are added to the context and the
/// client identity replaced; a failure changes nothing.
fn cb_load_cert(ctx: &mut CpuContext) -> HleResult {
    let (id, num, list, cert, key) = (ctx.rdi() as i32, ctx.rsi() as i32, ctx.rdx(), ctx.rcx(), ctx.r8());
    if num < 0 || (num != 0 && list == 0) || ((cert == 0) != (key == 0)) {
        return load_failure(&format!("invalid arguments (num {num}, list {list:#x}, cert {cert:#x}, key {key:#x})"), SSL_ERROR_INVALID_VALUE);
    }
    let Some(pool) = ssl().context(id).map(|c| c.pool.clone()) else {
        return load_failure(&format!("unknown context {id:#x}"), SSL_ERROR_INVALID_ID);
    };
    let mut chains = Vec::new();
    for i in 0..num as u64 {
        let Some(data) = ctx.read_u64(list + 8 * i) else {
            return ERR_MEMORY_FAULT;
        };
        if data == 0 {
            return load_failure("missing certificate descriptor", SSL_ERROR_INVALID_VALUE);
        }
        let bytes = match read_data(ctx, data) {
            Ok(b) => b,
            Err(e) => return e,
        };
        let certs = match parse_certificates(bytes) {
            Ok(c) => c,
            Err(what) => return load_failure(what, SSL_ERROR_INVALID_FORMAT),
        };
        let charge = match charge(&pool, der_size(&certs), "sceSslLoadCert") {
            Ok(c) => c,
            Err(e) => return e,
        };
        chains.push((certs, charge));
    }
    let identity = if cert != 0 {
        let (cert_bytes, key_bytes) = match (read_data(ctx, cert), read_data(ctx, key)) {
            (Ok(c), Ok(k)) => (c, k),
            (Err(e), _) | (_, Err(e)) => return e,
        };
        let certs = match parse_certificates(cert_bytes) {
            Ok(c) => c,
            Err(what) => return load_failure(what, SSL_ERROR_INVALID_FORMAT),
        };
        let key = match parse_key(key_bytes) {
            Ok(k) => k,
            Err(what) => return load_failure(what, SSL_ERROR_INVALID_FORMAT),
        };
        if !certs[0].public_key().is_ok_and(|p| p.public_eq(&key)) {
            return load_failure("client certificate and private key do not match", SSL_ERROR_MISMATCH_PUBLIC_KEYS);
        }
        let key_size = key.private_key_to_der().map_or(0, |d| d.len() as u64);
        let chain_charge = match charge(&pool, der_size(&certs), "sceSslLoadCert") {
            Ok(c) => c,
            Err(e) => return e,
        };
        let key_charge = match charge(&pool, key_size, "sceSslLoadCert") {
            Ok(c) => c,
            Err(e) => return e,
        };
        let chain = Arc::new(CertChain { certs, handles: Vec::new(), _charge: chain_charge });
        Some(Arc::new(Identity { chain, key, _charge: key_charge }))
    } else {
        None
    };
    // Guest handles for the new CA certificates.
    let mut loaded = Vec::new();
    for (certs, charge) in chains {
        let mut handles = Vec::with_capacity(certs.len());
        for _ in &certs {
            match crate::font::galloc(ctx, HANDLE_SIZE) {
                Some(h) => handles.push(h),
                None => {
                    handles.into_iter().for_each(crate::font::gfree);
                    loaded.into_iter().for_each(|c: Arc<CertChain>| c.handles.iter().copied().for_each(crate::font::gfree));
                    return load_failure("no guest memory for certificate handles", SSL_ERROR_OUT_OF_MEMORY);
                }
            }
        }
        loaded.push(Arc::new(CertChain { certs, handles, _charge: charge }));
    }
    let mut st = ssl();
    let Some(context) = st.context(id) else {
        loaded.into_iter().for_each(|c| c.handles.iter().copied().for_each(crate::font::gfree));
        return load_failure("the context was terminated during the load", SSL_ERROR_INVALID_ID);
    };
    for chain in loaded {
        for (i, &h) in chain.handles.iter().enumerate() {
            context.handles.insert(h, (Arc::downgrade(&chain), i));
        }
        context.authorities.push(chain);
    }
    if identity.is_some() {
        context.identity = identity;
    }
    0
}

/// sceSslUnloadCert(ctxId): drops the CA certificates and the client
/// identity; CA snapshots still held by the guest stay valid.
fn cb_unload_cert(ctx: &mut CpuContext) -> HleResult {
    let mut st = ssl();
    let Some(context) = st.context(ctx.rdi() as i32) else {
        return SSL_ERROR_INVALID_ID;
    };
    context.authorities.clear();
    context.identity = None;
    0
}

// ---------------------------------------------------------------------------
// CA queries
// ---------------------------------------------------------------------------

/// sceSslGetCaCerts(ctxId, SceSslCaCerts* caCerts): the DER bytes of every
/// loaded CA in one guest block (SceSslData array, then the bytes). An
/// unknown context or no CA clears the output and is NOT_FOUND.
fn cb_get_ca_certs(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rsi();
    if out == 0 {
        return SSL_ERROR_INVALID_ARG;
    }
    let mut st = ssl();
    let context = match st.context(ctx.rdi() as i32) {
        Some(c) if c.certificate_count() != 0 => c,
        _ => {
            return if ctx.write_bytes(out, &[0u8; CA_CERTS_SIZE]) { SSL_ERROR_NOT_FOUND } else { ERR_MEMORY_FAULT };
        }
    };
    let ders: Vec<Vec<u8>> = context.authorities.iter().flat_map(|c| c.certs.iter()).map(|c| c.to_der().unwrap_or_default()).collect();
    let count = ders.len();
    let total = count as u64 * DATA_SIZE + ders.iter().map(|d| d.len() as u64).sum::<u64>();
    let charge = match charge(&context.pool, total, "sceSslGetCaCerts") {
        Ok(c) => c,
        Err(e) => return e,
    };
    let Some(block) = crate::font::galloc(ctx, total) else {
        return SSL_ERROR_OUT_OF_MEMORY;
    };
    let mut table = Vec::with_capacity(total as usize);
    let mut next = block + count as u64 * DATA_SIZE;
    for d in &ders {
        table.extend_from_slice(&next.to_le_bytes());
        table.extend_from_slice(&(d.len() as u64).to_le_bytes());
        next += d.len() as u64;
    }
    ders.iter().for_each(|d| table.extend_from_slice(d));
    let mut header = [0u8; CA_CERTS_SIZE];
    header[0..8].copy_from_slice(&block.to_le_bytes());
    header[8..16].copy_from_slice(&(count as u64).to_le_bytes());
    header[16..24].copy_from_slice(&block.to_le_bytes());
    if !ctx.write_bytes(block, &table) || !ctx.write_bytes(out, &header) {
        crate::font::gfree(block);
        return ERR_MEMORY_FAULT;
    }
    context.ca_certs.insert(block, Snapshot { block, count, _chains: Vec::new(), _charge: charge });
    0
}

/// sceSslFreeCaCerts(ctxId, SceSslCaCerts* caCerts): releases an array
/// from sceSslGetCaCerts and clears the structure (any other contents are
/// only cleared).
fn cb_free_ca_certs(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rsi();
    if out == 0 {
        return SSL_ERROR_INVALID_ARG;
    }
    let Some(raw) = ctx.read_bytes(out, CA_CERTS_SIZE) else {
        return ERR_MEMORY_FAULT;
    };
    let (certs, num, pool) = (
        u64::from_le_bytes(raw[0..8].try_into().unwrap()),
        u64::from_le_bytes(raw[8..16].try_into().unwrap()),
        u64::from_le_bytes(raw[16..24].try_into().unwrap()),
    );
    {
        let mut st = ssl();
        if let Some(context) = st.context(ctx.rdi() as i32) {
            if let Some(s) = context.ca_certs.get(&pool) {
                if certs != pool || num != s.count as u64 {
                    log::warn!("[ssl] sceSslFreeCaCerts: the array at {pool:#x} does not match the one handed out");
                    return SSL_ERROR_INVALID_VALUE;
                }
                context.ca_certs.remove(&pool);
            }
        }
    }
    if ctx.write_bytes(out, &[0u8; CA_CERTS_SIZE]) { 0 } else { ERR_MEMORY_FAULT }
}

/// sceSslGetCaList(ctxId, SceSslCaList* caList): the handles of every
/// loaded CA certificate; no CA clears the list and is NOT_FOUND.
fn cb_get_ca_list(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rsi();
    if out == 0 {
        return SSL_ERROR_INVALID_ARG;
    }
    let mut st = ssl();
    let Some(context) = st.context(ctx.rdi() as i32) else {
        return SSL_ERROR_INVALID_ID;
    };
    let count = context.certificate_count();
    if count == 0 {
        return if ctx.write_bytes(out, &[0u8; CA_LIST_SIZE]) { SSL_ERROR_NOT_FOUND } else { ERR_MEMORY_FAULT };
    }
    let size = count as u64 * 8;
    let charge = match charge(&context.pool, size, "sceSslGetCaList") {
        Ok(c) => c,
        Err(e) => return e,
    };
    let Some(block) = crate::font::galloc(ctx, size) else {
        return SSL_ERROR_OUT_OF_MEMORY;
    };
    let pointers: Vec<u8> = context.authorities.iter().flat_map(|c| c.handles.iter()).flat_map(|h| h.to_le_bytes()).collect();
    let mut list = [0u8; CA_LIST_SIZE];
    list[0..8].copy_from_slice(&block.to_le_bytes());
    list[8..12].copy_from_slice(&(count as u32).to_le_bytes());
    if !ctx.write_bytes(block, &pointers) || !ctx.write_bytes(out, &list) {
        crate::font::gfree(block);
        return ERR_MEMORY_FAULT;
    }
    let chains = context.authorities.clone();
    context.ca_lists.insert(block, Snapshot { block, count, _chains: chains, _charge: charge });
    0
}

/// sceSslFreeCaList(ctxId, SceSslCaList* caList): releases a list from
/// sceSslGetCaList; an empty list is accepted, any other is refused.
fn cb_free_ca_list(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rsi();
    if out == 0 {
        return SSL_ERROR_INVALID_ARG;
    }
    let mut st = ssl();
    let Some(context) = st.context(ctx.rdi() as i32) else {
        return SSL_ERROR_INVALID_ID;
    };
    let (Some(certs), Some(num)) = (ctx.read_u64(out), ctx.read_u32(out + 8)) else {
        return ERR_MEMORY_FAULT;
    };
    if certs == 0 && num == 0 {
        return 0;
    }
    if !context.ca_lists.get(&certs).is_some_and(|s| s.count as u64 == num as u64) {
        log::warn!("[ssl] sceSslFreeCaList: list {certs:#x} ({num}) was not handed out by this context");
        return SSL_ERROR_INVALID_VALUE;
    }
    context.ca_lists.remove(&certs);
    if ctx.write_bytes(out, &[0u8; CA_LIST_SIZE]) { 0 } else { ERR_MEMORY_FAULT }
}

// ---------------------------------------------------------------------------
// Certificate readers
// ---------------------------------------------------------------------------

/// The certificate of `handle` in context `id`.
fn certificate(id: i32, handle: u64, function: &str) -> Result<X509, HleResult> {
    let mut st = ssl();
    let Some(context) = st.context(id) else {
        log::warn!("[ssl] {function}: unknown context {id:#x}");
        return Err(SSL_ERROR_INVALID_ID);
    };
    context.certificate(handle, function)
}

/// Microseconds since 0001-01-01 of an ASN.1 time.
fn time_ticks(t: &openssl::asn1::Asn1TimeRef) -> Option<u64> {
    const UNIX_EPOCH_DAYS: i64 = 719_162;
    let epoch = openssl::asn1::Asn1Time::from_unix(0).ok()?;
    let d = epoch.diff(t).ok()?;
    let seconds = (UNIX_EPOCH_DAYS + d.days as i64) * 86_400 + d.secs as i64;
    u64::try_from(seconds).ok().map(|s| s * 1_000_000)
}

fn validity_tick(ctx: &mut CpuContext, after: bool) -> HleResult {
    let function = if after { "sceSslGetNotAfter" } else { "sceSslGetNotBefore" };
    let out = ctx.rdx();
    if out == 0 {
        return SSL_ERROR_INVALID_ARG;
    }
    let cert = match certificate(ctx.rdi() as i32, ctx.rsi(), function) {
        Ok(c) => c,
        Err(e) => return e,
    };
    let time = if after { cert.not_after() } else { cert.not_before() };
    let Some(ticks) = time_ticks(time) else {
        log::warn!("[ssl] {function}: unreadable certificate time");
        return SSL_ERROR_INVALID_CERT;
    };
    if ctx.write_u64(out, ticks) { 0 } else { ERR_MEMORY_FAULT }
}

/// sceSslGetNotBefore(ctxId, const SceSslCert*, SceRtcTick* begin)
fn cb_get_not_before(ctx: &mut CpuContext) -> HleResult {
    validity_tick(ctx, false)
}

/// sceSslGetNotAfter(ctxId, const SceSslCert*, SceRtcTick* limit)
fn cb_get_not_after(ctx: &mut CpuContext) -> HleResult {
    validity_tick(ctx, true)
}

/// Fingerprint hash types: MD5, SHA-1, SHA-224, SHA-256, SHA-384, SHA-512.
fn fingerprint_digest(kind: i32) -> Option<MessageDigest> {
    Some(match kind {
        0 => MessageDigest::md5(),
        1 => MessageDigest::sha1(),
        2 => MessageDigest::sha224(),
        3 => MessageDigest::sha256(),
        4 => MessageDigest::sha384(),
        5 => MessageDigest::sha512(),
        _ => return None,
    })
}

/// sceSslGetFingerprint(ctxId, const SceSslCert*, int hashType, uint8_t* out):
/// the digest of the certificate's DER encoding.
fn cb_get_fingerprint(ctx: &mut CpuContext) -> HleResult {
    let (kind, out) = (ctx.rdx() as i32, ctx.rcx());
    if out == 0 {
        return SSL_ERROR_INVALID_ARG;
    }
    let Some(md) = fingerprint_digest(kind) else {
        return SSL_ERROR_INVALID_ARG;
    };
    let cert = match certificate(ctx.rdi() as i32, ctx.rsi(), "sceSslGetFingerprint") {
        Ok(c) => c,
        Err(e) => return e,
    };
    let Ok(digest) = cert.digest(md) else {
        return SSL_ERROR_INVALID_ARG;
    };
    if ctx.write_bytes(out, &digest) { 0 } else { ERR_MEMORY_FAULT }
}

/// sceSslGetSerialNumber(ctxId, const SceSslCert*, uint8_t* sboData,
/// size_t* sboLen): a NULL buffer or zero capacity queries the length; a
/// smaller buffer is INVALID_ARG with the length stored.
fn cb_get_serial_number(ctx: &mut CpuContext) -> HleResult {
    let (out, len_addr) = (ctx.rdx(), ctx.rcx());
    if len_addr == 0 {
        return SSL_ERROR_INVALID_ARG;
    }
    let cert = match certificate(ctx.rdi() as i32, ctx.rsi(), "sceSslGetSerialNumber") {
        Ok(c) => c,
        Err(e) => return e,
    };
    let serial = serial_bytes(&cert);
    let Some(capacity) = ctx.read_u64(len_addr) else {
        return ERR_MEMORY_FAULT;
    };
    // The length is written for a size query and a short buffer only.
    if out == 0 || capacity < serial.len() as u64 {
        if !ctx.write_u64(len_addr, serial.len() as u64) {
            return ERR_MEMORY_FAULT;
        }
        return if out == 0 || capacity == 0 { 0 } else { SSL_ERROR_INVALID_ARG };
    }
    if ctx.write_bytes(out, &serial) { 0 } else { ERR_MEMORY_FAULT }
}

/// The serial number's content octets as encoded (a leading zero kept).
pub(crate) fn serial_bytes(cert: &X509) -> Vec<u8> {
    let Ok(n) = cert.serial_number().to_bn() else {
        return Vec::new();
    };
    let mut bytes = n.to_vec();
    if bytes.is_empty() || bytes[0] & 0x80 != 0 {
        bytes.insert(0, 0);
    }
    bytes
}

/// The OID names the console reports in long form.
fn console_oid_name(short: &str) -> &str {
    match short {
        "CN" => "commonName",
        "C" => "countryName",
        "L" => "localityName",
        "ST" => "stateOrProvinceName",
        "O" => "organizationName",
        "OU" => "organizationalUnitName",
        "SN" => "surname",
        "GN" => "givenName",
        "DC" => "domainComponent",
        "UID" => "userId",
        other => other,
    }
}

/// (OID name as the console gives it, or dotted OID; raw value) of each
/// name entry.
fn name_entries(name: &X509NameRef) -> Vec<(String, Vec<u8>)> {
    name.entries()
        .map(|e| {
            let oid = e.object().nid().short_name().map(|s| console_oid_name(s).to_owned()).unwrap_or_else(|_| e.object().to_string());
            (oid, e.data().as_slice().to_vec())
        })
        .collect()
}

/// sceSslGetSubjectName / sceSslGetIssuerName(ctxId, const SceSslCert*) ->
/// SceSslCertName* (NULL on failure), held until sceSslFreeSslCertName or
/// sceSslTerm.
fn cert_name(ctx: &mut CpuContext, subject: bool) -> HleResult {
    let function = if subject { "sceSslGetSubjectName" } else { "sceSslGetIssuerName" };
    let id = ctx.rdi() as i32;
    let cert = match certificate(id, ctx.rsi(), function) {
        Ok(c) => c,
        Err(e) => {
            log::warn!("[ssl] {function}: certificate {:#x} refused ({e:#x})", ctx.rsi());
            return 0;
        }
    };
    let entries = name_entries(if subject { cert.subject_name() } else { cert.issuer_name() });
    let Some(handle) = crate::font::galloc(ctx, HANDLE_SIZE) else {
        return 0;
    };
    let mut st = ssl();
    match st.context(id) {
        Some(context) => {
            context.names.insert(handle, CertName { entries });
            handle as HleResult
        }
        None => {
            crate::font::gfree(handle);
            0
        }
    }
}

fn cb_get_subject_name(ctx: &mut CpuContext) -> HleResult {
    cert_name(ctx, true)
}

fn cb_get_issuer_name(ctx: &mut CpuContext) -> HleResult {
    cert_name(ctx, false)
}

/// Run `f` on a name of context `id`: INVALID_ARG for NULL, INVALID_ID for
/// an unknown context, INVALID_CERT for an unknown name.
fn with_name<R>(id: i32, name: u64, function: &str, f: impl FnOnce(&mut SslContext) -> R) -> Result<R, HleResult> {
    if name == 0 {
        return Err(SSL_ERROR_INVALID_ARG);
    }
    let mut st = ssl();
    let Some(context) = st.context(id) else {
        log::warn!("[ssl] {function}: unknown context {id:#x}");
        return Err(SSL_ERROR_INVALID_ID);
    };
    if !context.names.contains_key(&name) {
        log::warn!("[ssl] {function}: {name:#x} is not a certificate name of the context");
        return Err(SSL_ERROR_INVALID_CERT);
    }
    Ok(f(context))
}

/// sceSslGetNameEntryCount(ctxId, const SceSslCertName*)
fn cb_get_name_entry_count(ctx: &mut CpuContext) -> HleResult {
    let name = ctx.rsi();
    match with_name(ctx.rdi() as i32, name, "sceSslGetNameEntryCount", |c| c.names[&name].entries.len()) {
        Ok(n) => n as HleResult,
        Err(e) => e,
    }
}

/// NUL-terminated copy of `text` cut to `capacity - 1` bytes.
fn truncated(text: &[u8], capacity: u64) -> Vec<u8> {
    let n = text.len().min(capacity.saturating_sub(1) as usize);
    let mut b = text[..n].to_vec();
    b.push(0);
    b
}

/// sceSslGetNameEntryInfo(ctxId, const SceSslCertName*, int entryNum,
/// char* oidname, size_t maxOidnameLen, char* value, size_t maxValueLen,
/// size_t* valueLen): strings are cut to their buffers; valueLen gets the
/// full value length.
fn cb_get_name_entry_info(ctx: &mut CpuContext) -> HleResult {
    let (name, index, oid_out, oid_max, value_out) = (ctx.rsi(), ctx.rdx() as i32, ctx.rcx(), ctx.r8(), ctx.r9());
    let (value_max, len_out) = (stack_arg(ctx, 0), stack_arg(ctx, 1));
    let entry = match with_name(ctx.rdi() as i32, name, "sceSslGetNameEntryInfo", |c| {
        usize::try_from(index).ok().and_then(|i| c.names[&name].entries.get(i).cloned())
    }) {
        Ok(Some(e)) => e,
        Ok(None) => return SSL_ERROR_INVALID_ARG,
        Err(e) => return e,
    };
    let ok = (oid_out == 0 || oid_max == 0 || ctx.write_bytes(oid_out, &truncated(entry.0.as_bytes(), oid_max)))
        && (value_out == 0 || value_max == 0 || ctx.write_bytes(value_out, &truncated(&entry.1, value_max)))
        && (len_out == 0 || ctx.write_u64(len_out, entry.1.len() as u64));
    if ok { 0 } else { ERR_MEMORY_FAULT }
}

/// sceSslFreeSslCertName(ctxId, SceSslCertName*)
fn cb_free_cert_name(ctx: &mut CpuContext) -> HleResult {
    let name = ctx.rsi();
    match with_name(ctx.rdi() as i32, name, "sceSslFreeSslCertName", |c| c.names.remove(&name)) {
        Ok(_) => {
            crate::font::gfree(name);
            0
        }
        Err(e) => e,
    }
}

/// sceSslGetPem(ctxId, const SceSslCert*, char* pem, size_t* pemSize): not
/// supported; the certificate is still checked.
/// sceSslGetPem(ctxId, cert, char* buf, size_t* bufLen): the certificate as
/// PEM text with its NUL. A null buffer asks for the size; a buffer too
/// small is left untouched.
fn cb_get_pem(ctx: &mut CpuContext) -> HleResult {
    let (buf, len_addr) = (ctx.rdx(), ctx.rcx());
    if len_addr == 0 {
        return SSL_ERROR_INVALID_ARG;
    }
    let cert = match certificate(ctx.rdi() as i32, ctx.rsi(), "sceSslGetPem") {
        Ok(c) => c,
        Err(e) => return e,
    };
    let Ok(mut pem) = cert.to_pem() else {
        return SSL_ERROR_INVALID_CERT;
    };
    pem.push(0);
    let required = pem.len() as u64;
    if buf == 0 {
        return if ctx.write_u64(len_addr, required) { 0 } else { ERR_MEMORY_FAULT };
    }
    let Some(capacity) = ctx.read_u64(len_addr) else {
        return ERR_MEMORY_FAULT;
    };
    if capacity < required {
        log::warn!("[ssl] sceSslGetPem: buffer of {capacity} bytes for {required}");
        return SSL_ERROR_INVALID_ARG;
    }
    if !ctx.write_bytes(buf, &pem) || !ctx.write_u64(len_addr, required) {
        return ERR_MEMORY_FAULT;
    }
    0
}

// ---------------------------------------------------------------------------
// Connections
// ---------------------------------------------------------------------------

/// sceSslCreateConnection(ctxId, sockId, const char* host) -> connection id.
/// The socket is not checked until sceSslConnect.
fn cb_create_connection(ctx: &mut CpuContext) -> HleResult {
    let (id, socket, host) = (ctx.rdi() as i32, ctx.rsi() as i32, ctx.rdx());
    let Some(host) = (host != 0).then(|| ctx.read_cstr_bytes(host, MAX_HOST)).flatten() else {
        log::warn!("[ssl] sceSslCreateConnection: NULL or unreadable host");
        return SSL_ERROR_INVALID_VALUE;
    };
    let mut st = ssl();
    if st.context(id).is_none() {
        return SSL_ERROR_INVALID_ID;
    }
    let cid = st.next_connection;
    st.next_connection = cid.checked_add(1).unwrap_or(1);
    st.connections.insert(
        cid,
        SslConnection {
            context: id,
            socket,
            host: String::from_utf8_lossy(&host).into_owned(),
            verify_callback: 0,
            verify_arg: 0,
            verify_options: DEFAULT_VERIFY_OPTIONS,
            min_version: SSL_VERSION_MAX,
            alpn: Vec::new(),
            session: None,
        },
    );
    cid as HleResult
}

/// sceSslDeleteConnection(connId): no close_notify is sent and the socket
/// stays open; a connection whose context was terminated is INVALID_ID
/// (measured on a PS5, 13.40).
fn cb_delete_connection(ctx: &mut CpuContext) -> HleResult {
    let id = ctx.rdi() as i32;
    let mut st = ssl();
    if let Err(e) = st.connection(id, "sceSslDeleteConnection") {
        return e;
    }
    st.connections.remove(&id);
    0
}

/// Apply `f` to a live connection.
fn update_connection(id: i32, function: &str, f: impl FnOnce(&mut SslConnection)) -> HleResult {
    match ssl().connection(id, function) {
        Ok(c) => {
            f(c);
            0
        }
        Err(e) => e,
    }
}

/// sceSslSetVerifyCallback(connId, cbfunc, userArg)
fn cb_set_verify_callback(ctx: &mut CpuContext) -> HleResult {
    let (cb, arg) = (ctx.rsi(), ctx.rdx());
    update_connection(ctx.rdi() as i32, "sceSslSetVerifyCallback", |c| {
        c.verify_callback = cb;
        c.verify_arg = arg;
    })
}

/// Apply `f` to a live connection, which may refuse the value.
fn try_update_connection(id: i32, function: &str, f: impl FnOnce(&mut SslConnection) -> HleResult) -> HleResult {
    match ssl().connection(id, function) {
        Ok(c) => f(c),
        Err(e) => e,
    }
}

/// sceSslEnableVerifyOption(connId, option): the connection first, then
/// options outside 0x3F are INVALID_VALUE.
fn cb_enable_verify_option(ctx: &mut CpuContext) -> HleResult {
    let option = ctx.rsi() as u32;
    try_update_connection(ctx.rdi() as i32, "sceSslEnableVerifyOption", |c| {
        if option & !VERIFY_OPTIONS_MASK != 0 {
            return SSL_ERROR_INVALID_VALUE;
        }
        c.verify_options |= option;
        0
    })
}

/// sceSslDisableVerifyOption(connId, option)
fn cb_disable_verify_option(ctx: &mut CpuContext) -> HleResult {
    let option = ctx.rsi() as u32;
    try_update_connection(ctx.rdi() as i32, "sceSslDisableVerifyOption", |c| {
        if option & !VERIFY_OPTIONS_MASK != 0 {
            return SSL_ERROR_INVALID_VALUE;
        }
        c.verify_options &= !option;
        0
    })
}

/// sceSslSetMinSslVersion(connId, SceSslVersion): the connection first, then
/// a version outside 2 (SSL 3.0) .. 5 (TLS 1.2) is INVALID_VALUE.
fn cb_set_min_ssl_version(ctx: &mut CpuContext) -> HleResult {
    let version = ctx.rsi() as i32;
    try_update_connection(ctx.rdi() as i32, "sceSslSetMinSslVersion", |c| {
        if !(2..=SSL_VERSION_MAX).contains(&version) {
            log::warn!("[ssl] sceSslSetMinSslVersion: unknown version {version}");
            return SSL_ERROR_INVALID_VALUE;
        }
        c.min_version = version;
        0
    })
}

/// sceSslSetAlpn(connId, const char* protocols): an empty list is
/// BAD_LENGTH.
fn cb_set_alpn(ctx: &mut CpuContext) -> HleResult {
    const SSL_ERROR_BAD_LENGTH: HleResult = err(0x8095_1773);
    let Some(protocols) = (ctx.rsi() != 0).then(|| ctx.read_cstr_bytes(ctx.rsi(), MAX_HOST)).flatten() else {
        log::warn!("[ssl] sceSslSetAlpn: NULL protocol list");
        return SSL_ERROR_INVALID_VALUE;
    };
    try_update_connection(ctx.rdi() as i32, "sceSslSetAlpn", |c| {
        if protocols.is_empty() {
            return SSL_ERROR_BAD_LENGTH;
        }
        c.alpn = protocols;
        0
    })
}

/// sceSslGetAlpnSelected(connId) -> const char*: ALPN is never offered,
/// so no protocol is selected (NULL).
fn cb_get_alpn_selected(ctx: &mut CpuContext) -> HleResult {
    let _ = ssl().connection(ctx.rdi() as i32, "sceSslGetAlpnSelected");
    0
}

/// A TLS client for `host` with the context's CAs and client identity.
fn client(authorities: &[Arc<CertChain>], identity: Option<&Identity>, host: &str) -> Result<openssl::ssl::Ssl, openssl::error::ErrorStack> {
    let mut b = SslConnector::builder(SslMethod::tls_client())?;
    let mut store = X509StoreBuilder::new()?;
    for cert in authorities.iter().flat_map(|c| c.certs.iter()) {
        store.add_cert(cert.clone())?;
    }
    b.set_cert_store(store.build());
    b.set_verify(SslVerifyMode::PEER);
    b.set_max_proto_version(Some(SslVersion::TLS1_2))?;
    b.set_options(SslOptions::IGNORE_UNEXPECTED_EOF);
    if let Some(id) = identity {
        b.set_certificate(&id.chain.certs[0])?;
        for extra in &id.chain.certs[1..] {
            b.add_extra_chain_cert(extra.clone())?;
        }
        b.set_private_key(&id.key)?;
    }
    b.build().configure()?.into_ssl(host)
}

/// SCE_SSL_ERROR for a failed TLS operation (logged; the console's codes
/// for these failures are unknown).
fn tls_error(stream: &SslStream<HostSocket>, e: &openssl::ssl::Error, function: &str) -> HleResult {
    let verify = stream.ssl().verify_result();
    let code = match e.code() {
        ErrorCode::WANT_READ => SSL_ERROR_WANT_POLLIN,
        ErrorCode::WANT_WRITE => SSL_ERROR_WANT_POLLOUT,
        _ if verify != X509VerifyResult::OK => match verify.as_raw() {
            openssl_sys::X509_V_ERR_HOSTNAME_MISMATCH => SSL_ERROR_CN_CHECK,
            openssl_sys::X509_V_ERR_CERT_HAS_EXPIRED => SSL_ERROR_NOT_AFTER_CHECK,
            openssl_sys::X509_V_ERR_CERT_NOT_YET_VALID => SSL_ERROR_NOT_BEFORE_CHECK,
            _ => SSL_ERROR_UNKNOWN_CA,
        },
        ErrorCode::SYSCALL | ErrorCode::ZERO_RETURN => SSL_ERROR_SOCKET_CLOSED,
        _ => SSL_ERROR_FATAL_ALERT,
    };
    log::warn!("[ssl] {function}: {e} (verify: {}) -> {code:#x}", verify.error_string());
    code
}

/// sceSslConnect(connId): the TLS handshake over the connection's socket.
fn cb_connect(ctx: &mut CpuContext) -> HleResult {
    let id = ctx.rdi() as i32;
    let (socket, ssl_client) = {
        let mut st = ssl();
        let (context_id, socket, host) = match st.connection(id, "sceSslConnect") {
            Ok(c) if c.session.is_some() => {
                log::warn!("[ssl] sceSslConnect: connection {id:#x} is already connected");
                return SSL_ERROR_BUSY;
            }
            Ok(c) if c.verify_callback != 0 => {
                log::warn!("[ssl] sceSslConnect: how the console calls the verify callback is unknown");
                return SSL_ERROR_INVALID_VALUE;
            }
            Ok(c) => (c.context, c.socket, c.host.clone()),
            Err(e) => return e,
        };
        let context = st.context(context_id).expect("live context");
        if context.authorities.is_empty() {
            log::warn!("[ssl] sceSslConnect: no CA certificate is loaded (no root bundle is shipped)");
            return SSL_ERROR_UNKNOWN_CA;
        }
        let Some(sock) = HostSocket::of(socket) else {
            log::warn!("[ssl] sceSslConnect: descriptor {socket} is not a socket");
            return SSL_ERROR_INVALID_VALUE;
        };
        match client(&context.authorities, context.identity.as_deref(), &host) {
            Ok(c) => (sock, c),
            Err(e) => {
                log::warn!("[ssl] sceSslConnect: client setup failed: {e}");
                return SSL_ERROR_FATAL_ALERT;
            }
        }
    };
    let mut stream = match SslStream::new(ssl_client, socket) {
        Ok(s) => s,
        Err(e) => {
            log::warn!("[ssl] sceSslConnect: {e}");
            return SSL_ERROR_FATAL_ALERT;
        }
    };
    if let Err(e) = stream.connect() {
        return tls_error(&stream, &e, "sceSslConnect");
    }
    let mut st = ssl();
    match st.connections.get_mut(&id) {
        Some(c) => {
            c.session = Some(Arc::new(Mutex::new(stream)));
            0
        }
        None => {
            log::warn!("[ssl] sceSslConnect: connection {id:#x} was deleted during the handshake");
            SSL_ERROR_INVALID_ID
        }
    }
}

/// The session of a connected connection.
fn session(id: i32, function: &str) -> Result<Session, HleResult> {
    let mut st = ssl();
    let c = st.connection(id, function)?;
    c.session.clone().ok_or_else(|| {
        log::warn!("[ssl] {function}: connection {id:#x} is not connected");
        SSL_ERROR_BROKEN
    })
}

/// sceSslRead(connId, buf, len, flags): what one TLS read delivers (flags
/// 0 or MsgFlagPartialRead); 0 at the end of the stream.
fn cb_read(ctx: &mut CpuContext) -> HleResult {
    let (id, buf, len, flags) = (ctx.rdi() as i32, ctx.rsi(), ctx.rdx(), ctx.rcx() as i32);
    if flags != 0 && flags != READ_PARTIAL {
        log::warn!("[ssl] sceSslRead: unsupported flags {flags:#x}");
        return SSL_ERROR_INVALID_VALUE;
    }
    if buf == 0 && len != 0 {
        log::warn!("[ssl] sceSslRead: NULL buffer");
        return SSL_ERROR_INVALID_VALUE;
    }
    let session = match session(id, "sceSslRead") {
        Ok(s) => s,
        Err(e) => return e,
    };
    let mut data = vec![0u8; len.min(i32::MAX as u64).min(MAX_TRANSFER) as usize];
    let mut stream = session.lock().unwrap_or_else(|p| p.into_inner());
    let n = match stream.ssl_read(&mut data) {
        Ok(n) => n,
        Err(e) if e.code() == ErrorCode::ZERO_RETURN => 0,
        Err(e) => return tls_error(&stream, &e, "sceSslRead"),
    };
    drop(stream);
    if n != 0 && !ctx.write_bytes(buf, &data[..n]) {
        return ERR_MEMORY_FAULT;
    }
    n as HleResult
}

/// sceSslWrite(connId, buf, len, flags): sends all of `len` (flags 0).
fn cb_write(ctx: &mut CpuContext) -> HleResult {
    let (id, buf, len, flags) = (ctx.rdi() as i32, ctx.rsi(), ctx.rdx(), ctx.rcx() as i32);
    if flags != 0 {
        log::warn!("[ssl] sceSslWrite: unsupported flags {flags:#x}");
        return SSL_ERROR_INVALID_VALUE;
    }
    if buf == 0 && len != 0 {
        log::warn!("[ssl] sceSslWrite: NULL buffer");
        return SSL_ERROR_INVALID_VALUE;
    }
    let session = match session(id, "sceSslWrite") {
        Ok(s) => s,
        Err(e) => return e,
    };
    let total = len.min(i32::MAX as u64);
    let mut stream = session.lock().unwrap_or_else(|p| p.into_inner());
    let mut written = 0u64;
    while written < total {
        let n = (total - written).min(MAX_TRANSFER);
        let Some(data) = ctx.read_bytes(buf + written, n as usize) else {
            return ERR_MEMORY_FAULT;
        };
        let mut done = 0;
        while done < data.len() {
            match stream.ssl_write(&data[done..]) {
                Ok(k) => done += k,
                Err(e) => return tls_error(&stream, &e, "sceSslWrite"),
            }
        }
        written += n;
    }
    written as HleResult
}

const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("hdpVEUDFW3s", "sceSslInit", cb_ssl_init),
    ("0K1yQ6Lv-Yc", "sceSslTerm", cb_ssl_term),
    ("viRXSHZYd0c", "sceSslClose", cb_ssl_close),
    ("TDfQqO-gMbY", "sceSslGetCaCerts", cb_get_ca_certs),
    ("qIvLs0gYxi0", "sceSslFreeCaCerts", cb_free_ca_certs),
    ("qOn+wm28wmA", "sceSslGetCaList", cb_get_ca_list),
    ("+DzXseDVkeI", "sceSslFreeCaList", cb_free_ca_list),
    ("UQ+3Qu7v3cA", "sceSslUnloadCert", cb_unload_cert),
    ("-PoIzr3PEk0", "sceSslGetMemoryPoolStats", cb_get_memory_pool_stats),
    ("dQReuBX9sD8", "sceSslGetSubjectName", cb_get_subject_name),
    ("7whYpYfHP74", "sceSslGetIssuerName", cb_get_issuer_name),
    ("R1ePzopYPYM", "sceSslGetNameEntryCount", cb_get_name_entry_count),
    ("7RBSTKGrmDA", "sceSslGetNameEntryInfo", cb_get_name_entry_info),
    ("RwXD8grHZHM", "sceSslFreeSslCertName", cb_free_cert_name),
    ("DOwXL+FQMEY", "sceSslGetSerialNumber", cb_get_serial_number),
    ("kLB5aGoUJXg", "sceSslGetPem", cb_get_pem),
    ("Ab7+DH+gYyM", "sceSslLoadCert", cb_load_cert),
    ("Eo0S65Jy28Q", "sceSslGetNotBefore", cb_get_not_before),
    ("xHpt6+2pGYk", "sceSslGetNotAfter", cb_get_not_after),
    ("brRtwGBu4A8", "sceSslGetFingerprint", cb_get_fingerprint),
    ("tuscfitnhEo", "sceSslCreateConnection", cb_create_connection),
    ("HJ1n138CQ2g", "sceSslDeleteConnection", cb_delete_connection),
    ("E4a-ahM57QQ", "sceSslSetVerifyCallback", cb_set_verify_callback),
    ("po1X86mgHDU", "sceSslEnableVerifyOption", cb_enable_verify_option),
    ("PwsHbErG+e8", "sceSslDisableVerifyOption", cb_disable_verify_option),
    ("QWSxBzf6lAg", "sceSslSetMinSslVersion", cb_set_min_ssl_version),
    ("TL86glUrmUw", "sceSslSetAlpn", cb_set_alpn),
    ("4O7+bRkRUe8", "sceSslGetAlpnSelected", cb_get_alpn_selected),
    ("zXvd6iNyfgc", "sceSslConnect", cb_connect),
    ("jltWpVKtetg", "sceSslRead", cb_read),
    ("iNjkt9Poblw", "sceSslWrite", cb_write),
];

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, cb) in EXPORTS {
        reg.register_fn(nid, name, LIB, cb);
    }
}

#[cfg(test)]
#[path = "ssl_tests.rs"]
mod tests;
