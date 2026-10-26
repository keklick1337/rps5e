// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceSsl tests: context bookkeeping, certificate loading and readers,
//! and TLS connections to a loopback server (reference GuestSsl,
//! GuestSslLoad and GuestSslTransport cases).

use super::*;
use crate::context::CpuRegister::*;

const ROOT_PEM: &str = "-----BEGIN CERTIFICATE-----
MIIBxDCCAWqgAwIBAgIDAIChMAoGCCqGSM49BAMCMEAxCzAJBgNVBAYTAkdCMRQw
EgYDVQQKDAtycHM1ZSBUZXN0czEbMBkGA1UEAwwScnBzNWUgZml4dHVyZSByb290
MB4XDTI2MTAwODIyMTQwMloXDTM2MTAwNTIyMTQwMlowQDELMAkGA1UEBhMCR0Ix
FDASBgNVBAoMC3JwczVlIFRlc3RzMRswGQYDVQQDDBJycHM1ZSBmaXh0dXJlIHJv
b3QwWTATBgcqhkjOPQIBBggqhkjOPQMBBwNCAAQYF7C4jfsI4TWzfG3XLnaTGgNb
W8uXQjNIiUiMrIuuPzSSNlapc5HpfhOHxzIE9V5CQeHQIT5ONm0qkjcIv8fpo1Mw
UTAdBgNVHQ4EFgQUVmiVtNwaphUy1PoEPfuD3KpoDNwwHwYDVR0jBBgwFoAUVmiV
tNwaphUy1PoEPfuD3KpoDNwwDwYDVR0TAQH/BAUwAwEB/zAKBggqhkjOPQQDAgNI
ADBFAiBfE/jGfyxB06IvRFyzDQJ57GSir3AVKAfOyCD7eHFMdAIhAP9wxBHVhHoW
NqLVGTcm8w9KJAHo08/9cVeloUQTqYza
-----END CERTIFICATE-----
";
const CLIENT_PEM: &str = "-----BEGIN CERTIFICATE-----
MIIB2DCCAX+gAwIBAgIDEjRWMAoGCCqGSM49BAMCMEAxCzAJBgNVBAYTAkdCMRQw
EgYDVQQKDAtycHM1ZSBUZXN0czEbMBkGA1UEAwwScnBzNWUgZml4dHVyZSByb290
MB4XDTI2MTAwODIyMTQwMloXDTM2MTAwNTIyMTQwMlowZjELMAkGA1UEBhMCR0Ix
FDASBgNVBAoMC3JwczVlIFRlc3RzMSIwDgYDVQQLDAdQcmltYXJ5MBAGA1UECwwJ
U2Vjb25kYXJ5MR0wGwYDVQQDDBRycHM1ZSBmaXh0dXJlIGNsaWVudDBZMBMGByqG
SM49AgEGCCqGSM49AwEHA0IABHU2O/GGv+XNX8ScuU4TKKvvOKYw80QOda1lgFql
7fw/6MuRY1SCSVaHSshqpCruvMhK7T6Yr557XpskLep1axejQjBAMB0GA1UdDgQW
BBTBVJknMQt5DppwhOIf3b1niRS4RTAfBgNVHSMEGDAWgBRWaJW03BqmFTLU+gQ9
+4PcqmgM3DAKBggqhkjOPQQDAgNHADBEAiBCVnoJ9TSQai8yWGwHq3PGRcaB8q+o
P9rI5HK86yvf9QIgQR9QkDgu2WjW9sDO5U4VCVsly7upqGv3rUO9JRs1KOU=
-----END CERTIFICATE-----
";
const ROOT_KEY: &str = "-----BEGIN PRIVATE KEY-----
MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgHIXi+x3Z7ozqaKSg
w4mBVS67OC+pzhTpjT6EhQ9/V32hRANCAAQYF7C4jfsI4TWzfG3XLnaTGgNbW8uX
QjNIiUiMrIuuPzSSNlapc5HpfhOHxzIE9V5CQeHQIT5ONm0qkjcIv8fp
-----END PRIVATE KEY-----
";
const CLIENT_KEY: &str = "-----BEGIN PRIVATE KEY-----
MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgHoANtFzCHilXfZPd
GmO1G+NKG5sSUnUWmDrU29+IaQahRANCAAR1Njvxhr/lzV/EnLlOEyir7zimMPNE
DnWtZYBape38P+jLkWNUgklWh0rIaqQq7rzISu0+mK+ee16bJC3qdWsX
-----END PRIVATE KEY-----
";
const SERVER_PEM: &str = "-----BEGIN CERTIFICATE-----
MIIB9DCCAZugAwIBAgIDXhfhMAoGCCqGSM49BAMCMEAxCzAJBgNVBAYTAkdCMRQw
EgYDVQQKDAtycHM1ZSBUZXN0czEbMBkGA1UEAwwScnBzNWUgZml4dHVyZSByb290
MB4XDTI2MTAxNjExMjk1M1oXDTM2MTAxMzExMjk1M1owNzELMAkGA1UEBhMCR0Ix
FDASBgNVBAoMC3JwczVlIFRlc3RzMRIwEAYDVQQDDAlsb2NhbGhvc3QwWTATBgcq
hkjOPQIBBggqhkjOPQMBBwNCAASOWLxgY2XcCm5WcZKzafvQHUuJX1KtbMsNIbMU
yke+Sp9+LrTGU52se7eyb3nq5AVuGiBdmoW5QkBMFx00hkyBo4GMMIGJMBQGA1Ud
EQQNMAuCCWxvY2FsaG9zdDAMBgNVHRMBAf8EAjAAMA4GA1UdDwEB/wQEAwIHgDAT
BgNVHSUEDDAKBggrBgEFBQcDATAdBgNVHQ4EFgQUuuGJo54FNP2gUW5R7mLCBAz9
xkMwHwYDVR0jBBgwFoAUVmiVtNwaphUy1PoEPfuD3KpoDNwwCgYIKoZIzj0EAwID
RwAwRAIgWNG3mbycpcVUPi9ry69Zla33NmMRdeZpL0sf60t3pfICIHci1Ch2iO+P
Rwxyd7uQ8ejX2Qu5667hzQ/mMlnHdQOx
-----END CERTIFICATE-----
";
const SERVER_KEY: &str = "-----BEGIN PRIVATE KEY-----
MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQggjpMtOf9m8ZADfb/
g+KCUQ46CTPPM2bFyypaJZ3ZJq2hRANCAASOWLxgY2XcCm5WcZKzafvQHUuJX1Kt
bMsNIbMUyke+Sp9+LrTGU52se7eyb3nq5AVuGiBdmoW5QkBMFx00hkyB
-----END PRIVATE KEY-----
";

fn root_der() -> Vec<u8> {
    X509::from_pem(ROOT_PEM.as_bytes()).unwrap().to_der().unwrap()
}

/// Guest scratch memory with a bump allocator and call helpers.
struct Guest {
    ctx: CpuContext<'static>,
    next: u64,
    end: u64,
}

impl Guest {
    fn new(base: u64) -> Self {
        let vm: &'static ps5vm::VirtualMemory = Box::leak(Box::new(ps5vm::VirtualMemory::new()));
        vm.map(base, 0x10000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(vm);
        // Stack arguments live at the top of the scratch area.
        ctx.set(Rsp, base + 0xF000);
        Guest { ctx, next: base, end: base + 0xF000 }
    }

    fn alloc(&mut self, bytes: &[u8]) -> u64 {
        let a = self.next;
        self.next = (self.next + bytes.len() as u64 + 15) & !15;
        assert!(self.next <= self.end);
        assert!(self.ctx.write_bytes(a, bytes));
        a
    }

    fn zeroed(&mut self, n: usize) -> u64 {
        self.alloc(&vec![0; n])
    }

    fn cstr(&mut self, s: &str) -> u64 {
        let mut b = s.as_bytes().to_vec();
        b.push(0);
        self.alloc(&b)
    }

    /// A SceSslData for `bytes`.
    fn data(&mut self, bytes: &[u8]) -> u64 {
        let p = self.alloc(bytes);
        let mut d = p.to_le_bytes().to_vec();
        d.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        self.alloc(&d)
    }

    fn call(&mut self, cb: HleCallback, args: &[u64]) -> HleResult {
        for (r, v) in [Rdi, Rsi, Rdx, Rcx, R8, R9].iter().zip(args) {
            self.ctx.set(*r, *v);
        }
        for (i, v) in args.iter().skip(6).enumerate() {
            self.ctx.write_u64(self.end + 8 + 8 * i as u64, *v);
        }
        cb(&mut self.ctx)
    }

    fn init(&mut self, pool: u64) -> u64 {
        let id = self.call(cb_ssl_init, &[pool]);
        assert!(id > 0);
        id as u64
    }

    /// sceSslLoadCert with CA PEM/DER blobs and an optional identity.
    fn load(&mut self, id: u64, cas: &[&[u8]], identity: Option<(&[u8], &[u8])>) -> HleResult {
        let ptrs: Vec<u8> = cas.iter().map(|c| self.data(c)).flat_map(u64::to_le_bytes).collect();
        let list = if cas.is_empty() { 0 } else { self.alloc(&ptrs) };
        let (cert, key) = match identity {
            Some((c, k)) => (self.data(c), self.data(k)),
            None => (0, 0),
        };
        self.call(cb_load_cert, &[id, cas.len() as u64, list, cert, key])
    }

    /// (pool size, peak, current) from sceSslGetMemoryPoolStats.
    fn stats(&mut self, id: u64) -> (u64, u64, u64) {
        let out = self.zeroed(32);
        assert_eq!(self.call(cb_get_memory_pool_stats, &[id, out]), 0);
        let r = |o| self.ctx.read_u64(out + o).unwrap();
        let s = (r(0), r(8), r(16));
        assert!(s.2 <= s.1 && s.1 <= s.0 && self.ctx.read_u32(out + 24) == Some(0));
        s
    }

    fn in_use(&mut self, id: u64) -> u64 {
        self.stats(id).2
    }

    fn read(&self, addr: u64, n: usize) -> Vec<u8> {
        self.ctx.read_bytes(addr, n).unwrap()
    }
}

#[test]
fn nids_match_names() {
    for &(nid, name, _) in EXPORTS {
        assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
    }
}

#[test]
fn init_term() {
    let mut g = Guest::new(0x5E13_0000_0000);
    assert_eq!(g.call(cb_ssl_init, &[0]), 0x8002_0016u32 as i32 as i64);
    let id = g.init(0x4000);
    assert_eq!(g.call(cb_ssl_term, &[id]), 0);
    assert_eq!(g.call(cb_ssl_term, &[id]), SSL_ERROR_INVALID_ID);
}

/// An empty context: CA queries are NOT_FOUND and clear their outputs;
/// foreign lists are refused (reference GuestSsl, first half).
#[test]
fn empty_ca_store() {
    let mut g = Guest::new(0x5E13_0010_0000);
    let id = g.init(0x10000);
    assert_eq!(g.stats(id), (0x10000, 0, 0));
    // Measured on a console: NULL output first, then an unknown context
    // zeroes the 32-byte output.
    assert_eq!(g.call(cb_get_memory_pool_stats, &[id, 0]), SSL_ERROR_INVALID_VALUE);
    let stats = g.alloc(&[0x5A; 0x40]);
    assert_eq!(g.call(cb_get_memory_pool_stats, &[0x7777, stats]), SSL_ERROR_INVALID_ID);
    assert_eq!(g.read(stats, 0x40), [vec![0; 32], vec![0x5A; 32]].concat());

    for cb in [cb_get_ca_certs as HleCallback, cb_free_ca_certs, cb_get_ca_list, cb_free_ca_list] {
        assert_eq!(g.call(cb, &[id, 0]), SSL_ERROR_INVALID_ARG);
    }
    let marker = g.zeroed(16);
    let certs = g.alloc(&[marker.to_le_bytes(), 3u64.to_le_bytes(), marker.to_le_bytes()].concat());
    assert_eq!(g.call(cb_get_ca_certs, &[id, certs]), SSL_ERROR_NOT_FOUND);
    assert_eq!(g.read(certs, 24), vec![0; 24]);
    g.ctx.write_bytes(certs, &[marker.to_le_bytes(), 3u64.to_le_bytes(), marker.to_le_bytes()].concat());
    assert_eq!(g.call(cb_free_ca_certs, &[id, certs]), 0);
    assert_eq!(g.read(certs, 24), vec![0; 24]);
    // An unknown context has no CA either.
    assert_eq!(g.call(cb_get_ca_certs, &[0x7777, certs]), SSL_ERROR_NOT_FOUND);

    let list = g.zeroed(16);
    let set_list = |g: &mut Guest, certs: u64, num: u32| {
        g.ctx.write_u64(list, certs);
        g.ctx.write_u32(list + 8, num);
    };
    set_list(&mut g, marker, 3);
    assert_eq!(g.call(cb_get_ca_list, &[id, list]), SSL_ERROR_NOT_FOUND);
    assert_eq!((g.ctx.read_u64(list), g.ctx.read_u32(list + 8)), (Some(0), Some(0)));
    assert_eq!(g.call(cb_free_ca_list, &[id, list]), 0);
    for (certs, num) in [(marker, 1), (0, 1), (marker, 0)] {
        set_list(&mut g, certs, num);
        assert_eq!(g.call(cb_free_ca_list, &[id, list]), SSL_ERROR_INVALID_VALUE);
    }
    assert_eq!(g.call(cb_unload_cert, &[id]), 0);
    assert_eq!(g.call(cb_get_ca_list, &[id, list]), SSL_ERROR_NOT_FOUND);

    // Certificate readers refuse objects the library did not hand out.
    assert_eq!(g.call(cb_get_subject_name, &[id, marker]), 0);
    assert_eq!(g.call(cb_get_name_entry_count, &[id, marker]), SSL_ERROR_INVALID_CERT);
    let pem_len = g.zeroed(8);
    assert_eq!(g.call(cb_get_pem, &[id, 0, 0, pem_len]), SSL_ERROR_INVALID_VALUE);
    assert_eq!(g.call(cb_get_pem, &[id, marker, 0, 0]), SSL_ERROR_INVALID_ARG);

    assert_eq!(g.call(cb_ssl_term, &[id]), 0);
    assert_eq!(g.call(cb_get_ca_list, &[id, 0]), SSL_ERROR_INVALID_ARG);
    assert_eq!(g.call(cb_free_ca_list, &[id, 0]), SSL_ERROR_INVALID_ARG);
    set_list(&mut g, marker, 3);
    assert_eq!(g.call(cb_get_ca_list, &[id, list]), SSL_ERROR_INVALID_ID);
    assert_eq!(g.call(cb_free_ca_list, &[id, list]), SSL_ERROR_INVALID_ID);
    assert_eq!(g.call(cb_unload_cert, &[id]), SSL_ERROR_INVALID_ID);
}

/// Connection objects without a handshake (reference GuestSsl, second half).
#[test]
fn connection_bookkeeping() {
    let mut g = Guest::new(0x5E13_0020_0000);
    let id = g.init(0x10000);
    let host = g.cstr("example.com");
    assert_eq!(g.call(cb_create_connection, &[id, 0, 0]), SSL_ERROR_INVALID_VALUE);
    assert_eq!(g.call(cb_create_connection, &[id + 1000, 0, host]), SSL_ERROR_INVALID_ID);
    let conn = g.call(cb_create_connection, &[id, 0, host]);
    assert!(conn > 0 && conn as u64 != id);
    let conn = conn as u64;
    assert_eq!(g.call(cb_set_verify_callback, &[conn, 0xC0DE, 0x10]), 0);
    assert_eq!(g.call(cb_set_verify_callback, &[conn, 0, 0]), 0);
    assert_eq!(g.call(cb_enable_verify_option, &[conn, 0x1 | 0x4]), 0);
    assert_eq!(g.call(cb_disable_verify_option, &[conn, 0x4]), 0);
    assert_eq!(g.call(cb_disable_verify_option, &[conn, 0x5]), 0);
    // Measured on a PS5 (13.40): the connection is checked first, then
    // options outside 0x3F and versions outside 2..=5.
    assert_eq!(g.call(cb_enable_verify_option, &[conn, 0x40]), SSL_ERROR_INVALID_VALUE);
    assert_eq!(g.call(cb_disable_verify_option, &[conn, 0x80]), SSL_ERROR_INVALID_VALUE);
    assert_eq!(g.call(cb_enable_verify_option, &[0x7777, 0x40]), SSL_ERROR_INVALID_ID);
    assert_eq!(g.call(cb_set_min_ssl_version, &[0x7777, 9]), SSL_ERROR_INVALID_ID);
    for v in [2, 5] {
        assert_eq!(g.call(cb_set_min_ssl_version, &[conn, v]), 0);
    }
    for v in [-1i64 as u64, 0, 1, 6] {
        assert_eq!(g.call(cb_set_min_ssl_version, &[conn, v]), SSL_ERROR_INVALID_VALUE);
    }
    assert_eq!(ssl().connections[&(conn as i32)].verify_options, 0x20, "the default 0x25 less the disabled bits");
    assert_eq!(g.call(cb_get_alpn_selected, &[conn]), 0);
    let alpn = g.cstr("http/1.1");
    let empty = g.cstr("");
    assert_eq!(g.call(cb_set_alpn, &[conn, alpn]), 0);
    assert_eq!(g.call(cb_set_alpn, &[conn, empty]), 0x8095_1773u32 as i32 as i64);
    assert_eq!(g.call(cb_set_alpn, &[conn, 0]), SSL_ERROR_INVALID_VALUE);
    assert_eq!(g.call(cb_get_alpn_selected, &[conn]), 0);
    let buffer = g.zeroed(4);
    // No CA: no handshake; nothing to read or write before one.
    assert_eq!(g.call(cb_connect, &[conn]), SSL_ERROR_UNKNOWN_CA);
    assert_eq!(g.call(cb_read, &[conn, buffer, 4, 0]), SSL_ERROR_BROKEN);
    assert_eq!(g.call(cb_write, &[conn, buffer, 4, 0]), SSL_ERROR_BROKEN);

    let unknown = conn + 1000;
    for (cb, a) in [
        (cb_set_verify_callback as HleCallback, 0),
        (cb_enable_verify_option, 1),
        (cb_disable_verify_option, 1),
        (cb_set_min_ssl_version, 5),
        (cb_set_alpn, alpn),
    ] {
        assert_eq!(g.call(cb, &[unknown, a, 0]), SSL_ERROR_INVALID_ID);
    }
    assert_eq!(g.call(cb_delete_connection, &[unknown]), SSL_ERROR_INVALID_ID);

    // A terminated context leaves its connections unusable, deletion too
    // (measured on a PS5, 13.40).
    assert_eq!(g.call(cb_ssl_term, &[id]), 0);
    for (cb, a) in [(cb_set_verify_callback as HleCallback, 0), (cb_enable_verify_option, 1), (cb_set_min_ssl_version, 5), (cb_set_alpn, alpn)] {
        assert_eq!(g.call(cb, &[conn, a, 0]), SSL_ERROR_INVALID_ID);
    }
    assert_eq!(g.call(cb_create_connection, &[id, 0, host]), SSL_ERROR_INVALID_ID);
    assert_eq!(g.call(cb_delete_connection, &[conn]), SSL_ERROR_INVALID_ID);
    assert_eq!(g.call(cb_delete_connection, &[conn]), SSL_ERROR_INVALID_ID);
}

/// The CA snapshot of `id` holds exactly the fixture root.
fn check_root_snapshot(g: &Guest, certs: u64) {
    let der = root_der();
    assert_eq!(der.len(), 456);
    let array = g.ctx.read_u64(certs).unwrap();
    assert_eq!(g.ctx.read_u64(certs + 8), Some(1));
    assert_eq!(g.ctx.read_u64(certs + 16), Some(array));
    let (ptr, size) = (g.ctx.read_u64(array).unwrap(), g.ctx.read_u64(array + 8).unwrap());
    assert_eq!(size, der.len() as u64);
    assert_eq!(g.read(ptr, der.len()), der);
}

/// Reference GuestSslLoad Lifecycle.
fn lifecycle(base: u64) {
    let mut g = Guest::new(base);
    let id = g.init(1 << 20);
    let pem = g.data(ROOT_PEM.as_bytes());
    let list = g.alloc(&pem.to_le_bytes());
    assert_eq!(g.call(cb_load_cert, &[id, 1, list, 0, 0]), 0);
    // The guest copy may change: the context keeps its own.
    let text = g.ctx.read_u64(pem).unwrap();
    g.ctx.write_bytes(text, &vec![b'?'; ROOT_PEM.len()]);
    assert!(g.in_use(id) >= root_der().len() as u64);
    let bytes = g.zeroed(24);
    assert_eq!(g.call(cb_get_ca_certs, &[id, bytes]), 0);
    check_root_snapshot(&g, bytes);
    let handles = g.zeroed(16);
    assert_eq!(g.call(cb_get_ca_list, &[id, handles]), 0);
    assert_eq!(g.ctx.read_u32(handles + 8), Some(1));
    let cert = g.ctx.read_u64(g.ctx.read_u64(handles).unwrap()).unwrap();
    assert_ne!(cert, 0);
    let other = g.init(1 << 20);
    let empty = g.zeroed(16);
    assert_eq!(g.call(cb_get_ca_list, &[other, empty]), SSL_ERROR_NOT_FOUND);
    let foreign = g.alloc(&g.read(handles, 16));
    assert_eq!(g.call(cb_free_ca_list, &[other, foreign]), SSL_ERROR_INVALID_VALUE);
    assert_eq!(g.call(cb_unload_cert, &[id]), 0);
    assert_eq!(g.call(cb_get_ca_list, &[id, empty]), SSL_ERROR_NOT_FOUND);
    assert_eq!(g.read(empty, 16), vec![0; 16]);
    // Snapshots outlive the unload.
    check_root_snapshot(&g, bytes);
    assert_eq!(g.call(cb_free_ca_list, &[id, handles]), 0);
    assert_eq!(g.read(handles, 16), vec![0; 16]);
    assert_eq!(g.call(cb_free_ca_certs, &[id, bytes]), 0);
    assert_eq!(g.read(bytes, 24), vec![0; 24]);
    assert_eq!(g.in_use(id), 0);
    assert_eq!(g.call(cb_free_ca_list, &[id, foreign]), SSL_ERROR_INVALID_VALUE);
    assert_eq!(g.call(cb_ssl_term, &[id]), 0);
    assert_eq!(g.call(cb_ssl_term, &[other]), 0);
}

#[test]
fn load_lifecycle() {
    lifecycle(0x5E13_0030_0000);
    let workers: Vec<_> = (0..3u64)
        .map(|w| std::thread::spawn(move || (0..4u64).for_each(|n| lifecycle(0x5E13_0100_0000 + (w * 4 + n) * 0x10_0000))))
        .collect();
    workers.into_iter().for_each(|w| w.join().unwrap());
}

/// Reference GuestSslLoad AtomicFailures: a failed load changes nothing.
#[test]
fn load_failures_change_nothing() {
    let mut g = Guest::new(0x5E13_0040_0000);
    let id = g.init(1 << 20);
    let der = root_der();
    assert_eq!(g.load(id, &[&der], None), 0);
    let before = g.in_use(id);
    assert_eq!(g.load(id, &[&der, b"not a certificate"], None), SSL_ERROR_INVALID_FORMAT);
    assert_eq!(g.in_use(id), before);
    let bytes = g.zeroed(24);
    assert_eq!(g.call(cb_get_ca_certs, &[id, bytes]), 0);
    check_root_snapshot(&g, bytes);
    assert_eq!(g.call(cb_free_ca_certs, &[id, bytes]), 0);
    // DER with trailing bytes is refused.
    let mut trailing = der.clone();
    trailing.push(0);
    assert_eq!(g.load(id, &[&trailing], None), SSL_ERROR_INVALID_FORMAT);
    assert_eq!(g.call(cb_load_cert, &[id, -1i64 as u64, 0, 0, 0]), SSL_ERROR_INVALID_VALUE);
    assert_eq!(g.call(cb_load_cert, &[id, 1, 0, 0, 0]), SSL_ERROR_INVALID_VALUE);
    let absent = g.zeroed(8);
    assert_eq!(g.call(cb_load_cert, &[id, 1, absent, 0, 0]), SSL_ERROR_INVALID_VALUE);
    let cert = g.data(&der);
    assert_eq!(g.call(cb_load_cert, &[id, 0, 0, cert, 0]), SSL_ERROR_INVALID_VALUE);
    assert_eq!(g.in_use(id), before);
    assert_eq!(g.call(cb_ssl_term, &[id]), 0);
    assert_eq!(g.load(id, &[&der], None), SSL_ERROR_INVALID_ID);
    // A pool too small for the certificate.
    let tiny = g.init(32);
    assert_eq!(g.load(tiny, &[&der], None), SSL_ERROR_OUT_OF_MEMORY);
    assert_eq!(g.in_use(tiny), 0);
    assert_eq!(g.call(cb_ssl_term, &[tiny]), 0);
}

/// Reference GuestSslLoad ClientIdentity.
#[test]
fn client_identity() {
    let mut g = Guest::new(0x5E13_0050_0000);
    let id = g.init(1 << 20);
    assert_eq!(g.load(id, &[], Some((CLIENT_PEM.as_bytes(), CLIENT_KEY.as_bytes()))), 0);
    let before = g.in_use(id);
    assert!(before > 0);
    assert_eq!(g.load(id, &[], Some((CLIENT_PEM.as_bytes(), ROOT_KEY.as_bytes()))), SSL_ERROR_MISMATCH_PUBLIC_KEYS);
    assert_eq!(g.in_use(id), before);
    assert_eq!(g.call(cb_unload_cert, &[id]), 0);
    assert_eq!(g.in_use(id), 0);
    assert_eq!(g.call(cb_ssl_term, &[id]), 0);
}

/// Reference GuestSslLoad ConnectionLifetime.
#[test]
fn connections_outlive_loads_and_term() {
    let mut g = Guest::new(0x5E13_0060_0000);
    let id = g.init(1 << 20);
    let host = g.cstr("example.invalid");
    let conn = g.call(cb_create_connection, &[id, -1i64 as u64, host]) as u64;
    let other = g.init(1 << 20);
    assert!(conn as i64 > 0 && conn != id && conn != other);
    assert_eq!(g.load(id, &[ROOT_PEM.as_bytes()], None), 0);
    assert_eq!(g.call(cb_set_verify_callback, &[conn, 0, 0]), 0);
    let bytes = g.zeroed(24);
    assert_eq!(g.call(cb_get_ca_certs, &[id, bytes]), 0);
    check_root_snapshot(&g, bytes);
    let empty = g.zeroed(24);
    assert_eq!(g.call(cb_get_ca_certs, &[other, empty]), SSL_ERROR_NOT_FOUND);
    let stats = g.zeroed(32);
    assert_eq!(g.call(cb_get_memory_pool_stats, &[conn, stats]), SSL_ERROR_INVALID_ID);
    let retained = g.call(cb_create_connection, &[id, -1i64 as u64, host]) as u64;
    assert!(retained as i64 > 0 && retained != conn && retained != other);
    assert_eq!(g.call(cb_delete_connection, &[conn]), 0);
    assert_eq!(g.call(cb_unload_cert, &[id]), 0);
    check_root_snapshot(&g, bytes);
    assert_eq!(g.call(cb_set_verify_callback, &[retained, 0, 0]), 0);
    assert_eq!(g.call(cb_free_ca_certs, &[id, bytes]), 0);
    assert_eq!(g.in_use(id), 0);
    assert_eq!(g.call(cb_ssl_term, &[id]), 0);
    // Term drops the context's connections: a later context in the same
    // slot cannot revive them.
    assert!(!ssl().connections.contains_key(&(retained as i32)));
    assert_eq!(g.call(cb_set_verify_callback, &[retained, 0, 0]), SSL_ERROR_INVALID_ID);
    // A connection of a terminated context: INVALID_ID (PS5, 13.40).
    assert_eq!(g.call(cb_delete_connection, &[retained]), SSL_ERROR_INVALID_ID);
    assert_eq!(g.call(cb_ssl_close, &[retained]), SSL_ERROR_INVALID_ID);
    assert_eq!(g.call(cb_get_ca_certs, &[other, empty]), SSL_ERROR_NOT_FOUND);
    assert_eq!(g.call(cb_ssl_term, &[other]), 0);
}

/// Reference GuestSslLoad CertificateReaders.
#[test]
fn certificate_readers() {
    let mut g = Guest::new(0x5E13_0070_0000);
    let id = g.init(1 << 20);
    assert_eq!(g.load(id, &[ROOT_PEM.as_bytes()], None), 0);
    let handles = g.zeroed(16);
    assert_eq!(g.call(cb_get_ca_list, &[id, handles]), 0);
    let cert = g.ctx.read_u64(g.ctx.read_u64(handles).unwrap()).unwrap();
    let fake = g.zeroed(16);

    // Serial number: query, short buffer, copy.
    let len = g.zeroed(8);
    assert_eq!(g.call(cb_get_serial_number, &[id, cert, 0, len]), 0);
    assert_eq!(g.ctx.read_u64(len), Some(3));
    let short = g.zeroed(2);
    g.ctx.write_u64(len, 0);
    assert_eq!(g.call(cb_get_serial_number, &[id, cert, short, len]), 0);
    assert_eq!(g.ctx.read_u64(len), Some(3));
    g.ctx.write_u64(len, 2);
    assert_eq!(g.call(cb_get_serial_number, &[id, cert, short, len]), SSL_ERROR_INVALID_ARG);
    assert_eq!(g.ctx.read_u64(len), Some(3));
    // A copy that fits leaves the length alone (PS5, 13.40).
    let serial = g.zeroed(8);
    g.ctx.write_u64(len, 8);
    assert_eq!(g.call(cb_get_serial_number, &[id, cert, serial, len]), 0);
    assert_eq!((g.ctx.read_u64(len), g.read(serial, 3)), (Some(8), vec![0x00, 0x80, 0xA1]));
    assert_eq!(g.call(cb_get_serial_number, &[id, cert, serial, 0]), SSL_ERROR_INVALID_ARG);
    assert_eq!(g.call(cb_get_serial_number, &[id, 0, serial, len]), SSL_ERROR_INVALID_VALUE);
    assert_eq!(g.call(cb_get_serial_number, &[id, fake, serial, len]), SSL_ERROR_INVALID_CERT);
    assert_eq!(g.call(cb_get_serial_number, &[9999, cert, serial, len]), SSL_ERROR_INVALID_ID);

    // PEM text: size query, a buffer too small left untouched, the copy
    // with its NUL and nothing past it.
    let pem_len = g.zeroed(8);
    assert_eq!(g.call(cb_get_pem, &[id, cert, 0, pem_len]), 0);
    let need = ROOT_PEM.len() as u64 + 1;
    assert_eq!(g.ctx.read_u64(pem_len), Some(need));
    let small = g.zeroed(4);
    g.ctx.write_u64(pem_len, 2);
    assert_eq!(g.call(cb_get_pem, &[id, cert, small, pem_len]), SSL_ERROR_INVALID_ARG);
    assert_eq!(g.read(small, 4), vec![0; 4]);
    let text = g.zeroed(need as usize + 2);
    g.ctx.write_bytes(text, &vec![0x5A; need as usize + 2]);
    g.ctx.write_u64(pem_len, need + 2);
    assert_eq!(g.call(cb_get_pem, &[id, cert, text, pem_len]), 0);
    assert_eq!(g.ctx.read_u64(pem_len), Some(need));
    let mut want = ROOT_PEM.as_bytes().to_vec();
    want.extend_from_slice(&[0, 0x5A, 0x5A]);
    assert_eq!(g.read(text, need as usize + 2), want);
    assert_eq!(g.call(cb_get_pem, &[id, cert, text, 0]), SSL_ERROR_INVALID_ARG);
    assert_eq!(g.call(cb_get_pem, &[id, fake, text, pem_len]), SSL_ERROR_INVALID_CERT);
    assert_eq!(g.call(cb_get_pem, &[9999, cert, text, pem_len]), SSL_ERROR_INVALID_ID);

    // Subject and issuer names.
    let subject = g.call(cb_get_subject_name, &[id, cert]) as u64;
    let issuer = g.call(cb_get_issuer_name, &[id, cert]) as u64;
    assert!(subject != 0 && issuer != 0);
    for bad in [[id, 0], [id, fake], [9999, cert]] {
        assert_eq!(g.call(cb_get_subject_name, &bad), 0);
        assert_eq!(g.call(cb_get_issuer_name, &bad), 0);
    }
    assert_eq!(g.call(cb_get_name_entry_count, &[id, subject]), 3);
    assert_eq!(g.call(cb_get_name_entry_count, &[id, issuer]), 3);
    assert_eq!(g.call(cb_get_name_entry_count, &[id, 0]), SSL_ERROR_INVALID_ARG);
    assert_eq!(g.call(cb_get_name_entry_count, &[id, fake]), SSL_ERROR_INVALID_CERT);
    assert_eq!(g.call(cb_get_name_entry_count, &[9999, subject]), SSL_ERROR_INVALID_ID);
    let (oid, val, vlen) = (g.zeroed(32), g.zeroed(64), g.zeroed(8));
    for (i, (o, v)) in [("countryName", "GB"), ("organizationName", "rps5e Tests"), ("commonName", "rps5e fixture root")].iter().enumerate() {
        assert_eq!(g.call(cb_get_name_entry_info, &[id, subject, i as u64, oid, 32, val, 64, vlen]), 0);
        assert_eq!(g.ctx.read_cstring(oid, 32).as_deref(), Some(*o));
        assert_eq!(g.ctx.read_cstring(val, 64).as_deref(), Some(*v));
        assert_eq!(g.ctx.read_u64(vlen), Some(v.len() as u64));
    }
    let small = g.zeroed(4);
    assert_eq!(g.call(cb_get_name_entry_info, &[id, subject, 1, 0, 0, small, 4, vlen]), 0);
    assert_eq!((g.ctx.read_cstring(small, 4).as_deref(), g.ctx.read_u64(vlen)), (Some("rps"), Some(11)));
    for index in [-1i64 as u64, 3] {
        assert_eq!(g.call(cb_get_name_entry_info, &[id, subject, index, oid, 32, val, 64, vlen]), SSL_ERROR_INVALID_ARG);
    }
    assert_eq!(g.call(cb_get_name_entry_info, &[id, 0, 0, oid, 32, val, 64, vlen]), SSL_ERROR_INVALID_ARG);
    assert_eq!(g.call(cb_get_name_entry_info, &[id, fake, 0, oid, 32, val, 64, vlen]), SSL_ERROR_INVALID_CERT);
    assert_eq!(g.call(cb_get_name_entry_info, &[9999, subject, 0, oid, 32, val, 64, vlen]), SSL_ERROR_INVALID_ID);
    assert_eq!(g.call(cb_free_cert_name, &[id, subject]), 0);
    assert_eq!(g.call(cb_free_cert_name, &[id, issuer]), 0);
    assert_eq!(g.call(cb_free_cert_name, &[id, 0]), SSL_ERROR_INVALID_ARG);
    assert_eq!(g.call(cb_free_cert_name, &[id, subject]), SSL_ERROR_INVALID_CERT);
    assert_eq!(g.call(cb_free_cert_name, &[9999, subject]), SSL_ERROR_INVALID_ID);

    // Validity in microseconds since 0001-01-01.
    let tick = g.zeroed(8);
    assert_eq!(g.call(cb_get_not_before, &[id, cert, tick]), 0);
    assert_eq!(g.ctx.read_u64(tick), Some(63_927_094_442_000_000));
    assert_eq!(g.call(cb_get_not_after, &[id, cert, tick]), 0);
    assert_eq!(g.ctx.read_u64(tick), Some(64_242_454_442_000_000));
    for cb in [cb_get_not_before as HleCallback, cb_get_not_after] {
        assert_eq!(g.call(cb, &[id, cert, 0]), SSL_ERROR_INVALID_ARG);
        assert_eq!(g.call(cb, &[id, 0, tick]), SSL_ERROR_INVALID_VALUE);
        assert_eq!(g.call(cb, &[id, fake, tick]), SSL_ERROR_INVALID_CERT);
        assert_eq!(g.call(cb, &[9999, cert, tick]), SSL_ERROR_INVALID_ID);
    }

    // Fingerprints of the DER encoding.
    let digest = g.zeroed(64);
    assert_eq!(g.call(cb_get_fingerprint, &[id, cert, 0, digest]), 0);
    assert_eq!(g.read(digest, 16), [0xe7, 0x8b, 0xcd, 0xe1, 0x02, 0x86, 0x25, 0x44, 0x5c, 0x51, 0xcf, 0x9d, 0x4b, 0x3e, 0x2d, 0xc4]);
    assert_eq!(g.call(cb_get_fingerprint, &[id, cert, 1, digest]), 0);
    assert_eq!(
        g.read(digest, 20),
        [0x8e, 0x76, 0xe4, 0x0b, 0xe2, 0x47, 0x79, 0x19, 0x1f, 0x53, 0x56, 0xa7, 0xe5, 0x2e, 0x3b, 0x79, 0x67, 0x31, 0x7c, 0x23]
    );
    assert_eq!(g.call(cb_get_fingerprint, &[id, cert, 3, digest]), 0);
    assert_eq!(
        g.read(digest, 32),
        [
            0xd8, 0x7e, 0x33, 0x1b, 0xe1, 0x6a, 0x30, 0xbc, 0x1e, 0xec, 0xb0, 0xe1, 0xd2, 0xff, 0x52, 0x43, 0x07, 0xed, 0x7d, 0xcd, 0xcd, 0x55, 0x60, 0xa9,
            0xf9, 0x39, 0x79, 0x37, 0x2e, 0x94, 0xda, 0x16
        ]
    );
    assert_eq!(g.call(cb_get_fingerprint, &[id, cert, 3, 0]), SSL_ERROR_INVALID_ARG);
    for kind in [-1i64 as u64, 99] {
        assert_eq!(g.call(cb_get_fingerprint, &[id, cert, kind, digest]), SSL_ERROR_INVALID_ARG);
    }
    assert_eq!(g.call(cb_get_fingerprint, &[id, 0, 3, digest]), SSL_ERROR_INVALID_VALUE);
    assert_eq!(g.call(cb_get_fingerprint, &[id, fake, 3, digest]), SSL_ERROR_INVALID_CERT);
    assert_eq!(g.call(cb_get_fingerprint, &[9999, cert, 3, digest]), SSL_ERROR_INVALID_ID);

    // After term nothing resolves.
    assert_eq!(g.call(cb_free_ca_list, &[id, handles]), 0);
    assert_eq!(g.call(cb_ssl_term, &[id]), 0);
    assert_eq!(g.call(cb_get_not_before, &[id, cert, tick]), SSL_ERROR_INVALID_ID);
    assert_eq!(g.call(cb_get_subject_name, &[id, cert]), 0);

    // An unloaded certificate is gone once no snapshot holds it.
    let id2 = g.init(1 << 20);
    assert_eq!(g.load(id2, &[ROOT_PEM.as_bytes()], None), 0);
    assert_eq!(g.call(cb_get_ca_list, &[id2, handles]), 0);
    let cert2 = g.ctx.read_u64(g.ctx.read_u64(handles).unwrap()).unwrap();
    let subject2 = g.call(cb_get_subject_name, &[id2, cert2]) as u64;
    assert_ne!(subject2, 0);
    assert_eq!(g.call(cb_free_ca_list, &[id2, handles]), 0);
    assert_eq!(g.call(cb_unload_cert, &[id2]), 0);
    assert_eq!(g.call(cb_get_serial_number, &[id2, cert2, serial, len]), SSL_ERROR_INVALID_CERT);
    assert_eq!(g.call(cb_get_subject_name, &[id2, cert2]), 0);
    assert_eq!(g.call(cb_get_fingerprint, &[id2, cert2, 3, digest]), SSL_ERROR_INVALID_CERT);
    // A name outlives its certificate until the context goes.
    assert_eq!(g.call(cb_get_name_entry_count, &[id2, subject2]), 3);
    assert_eq!(g.call(cb_ssl_term, &[id2]), 0);
    assert_eq!(g.call(cb_get_name_entry_count, &[id2, subject2]), SSL_ERROR_INVALID_ID);
}

/// A loopback TLS server with the fixture server certificate that requires
/// a client certificate from the fixture root. Each accepted connection
/// either fails its handshake or answers "ping" with "pong" and a
/// close_notify.
fn serve(listener: std::net::TcpListener, outcomes: Vec<bool>) -> std::thread::JoinHandle<Result<(), String>> {
    use openssl::ssl::{SslAcceptor, SslFiletype};
    let _ = SslFiletype::PEM;
    std::thread::spawn(move || {
        let mut b = SslAcceptor::mozilla_intermediate_v5(SslMethod::tls()).map_err(|e| e.to_string())?;
        b.set_certificate(&X509::from_pem(SERVER_PEM.as_bytes()).unwrap()).unwrap();
        b.set_private_key(&PKey::private_key_from_pem(SERVER_KEY.as_bytes()).unwrap()).unwrap();
        let mut store = X509StoreBuilder::new().unwrap();
        store.add_cert(X509::from_pem(ROOT_PEM.as_bytes()).unwrap()).unwrap();
        b.set_cert_store(store.build());
        b.set_verify(SslVerifyMode::PEER | SslVerifyMode::FAIL_IF_NO_PEER_CERT);
        let acceptor = b.build();
        for succeed in outcomes {
            let (tcp, _) = listener.accept().map_err(|e| e.to_string())?;
            match acceptor.accept(tcp) {
                Ok(mut s) if succeed => {
                    let mut request = [0u8; 4];
                    s.read_exact(&mut request).map_err(|e| e.to_string())?;
                    if &request != b"ping" {
                        return Err("wrong request".into());
                    }
                    s.write_all(b"pong").map_err(|e| e.to_string())?;
                    s.shutdown().map_err(|e| e.to_string())?;
                }
                Ok(_) => return Err("a handshake meant to fail succeeded".into()),
                Err(e) if succeed => return Err(format!("handshake failed: {e}")),
                Err(_) => {}
            }
        }
        Ok(())
    })
}

/// Reference GuestSslTransport over libSceNet sockets.
#[test]
fn tls_over_loopback() {
    const PARALLEL: usize = 4;
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let server = serve(listener, [vec![false, true], vec![true; PARALLEL]].concat());
    let mut g = Guest::new(0x5E13_0080_0000);
    let localhost = g.cstr("localhost");

    let context = g.init(1 << 20);
    assert_eq!(g.load(context, &[ROOT_PEM.as_bytes()], Some((CLIENT_PEM.as_bytes(), CLIENT_KEY.as_bytes()))), 0);
    let loaded = g.in_use(context);
    let callback = g.call(cb_create_connection, &[context, -1i64 as u64, localhost]) as u64;
    assert_eq!(g.call(cb_set_verify_callback, &[callback, 0xC0DE, 0]), 0);
    assert_eq!(g.call(cb_connect, &[callback]), SSL_ERROR_INVALID_VALUE);
    assert_eq!(g.call(cb_delete_connection, &[callback]), 0);
    let closed = g.call(cb_create_connection, &[context, 1000, localhost]) as u64;
    assert_eq!(g.call(cb_connect, &[closed]), SSL_ERROR_INVALID_VALUE);
    assert_eq!(g.call(cb_delete_connection, &[closed]), 0);

    // The server's certificate names localhost, not example.com.
    let socket = crate::net::tests_connected_socket(port);
    let example = g.cstr("example.com");
    let mismatch = g.call(cb_create_connection, &[context, socket as u64, example]) as u64;
    assert_eq!(g.call(cb_connect, &[mismatch]), SSL_ERROR_CN_CHECK);
    let buffer = g.zeroed(16);
    assert_eq!(g.call(cb_read, &[mismatch, buffer, 1, 0]), SSL_ERROR_BROKEN);
    assert_eq!(g.call(cb_delete_connection, &[mismatch]), 0);
    assert!(crate::net::tests_close_socket(socket));

    let socket = crate::net::tests_connected_socket(port);
    let conn = g.call(cb_create_connection, &[context, socket as u64, localhost]) as u64;
    assert_eq!(g.call(cb_read, &[conn, buffer, 16, 0]), SSL_ERROR_BROKEN);
    assert_eq!(g.call(cb_write, &[conn, buffer, 16, 0]), SSL_ERROR_BROKEN);
    assert_eq!(g.call(cb_connect, &[conn]), 0);
    assert_eq!(g.call(cb_connect, &[conn]), SSL_ERROR_BUSY);
    assert_eq!(g.call(cb_write, &[conn, buffer, 1, 1]), SSL_ERROR_INVALID_VALUE);
    assert_eq!(g.call(cb_read, &[conn, buffer, 1, 3]), SSL_ERROR_INVALID_VALUE);
    let ping = g.alloc(b"ping");
    assert_eq!(g.call(cb_write, &[conn, ping, 4, 0]), 4);
    assert_eq!(g.call(cb_read, &[conn, buffer, 16, 0]), 4);
    assert_eq!(g.read(buffer, 4), b"pong");
    // close_notify ends the stream.
    assert_eq!(g.call(cb_read, &[conn, buffer, 16, READ_PARTIAL as u64]), 0);
    assert_eq!(g.call(cb_delete_connection, &[conn]), 0);
    assert_eq!(g.in_use(context), loaded);
    // The socket stays open after the connection is deleted.
    assert!(crate::net::tests_close_socket(socket));

    let vm = g.ctx.vm;
    let clients: Vec<_> = (0..PARALLEL as u64)
        .map(|i| {
            std::thread::spawn(move || {
                let mut ctx = CpuContext::new(vm);
                let base = 0x5E13_0080_8000 + i * 0x400;
                let socket = crate::net::tests_connected_socket(port);
                let call = |ctx: &mut CpuContext, cb: HleCallback, args: &[u64]| {
                    for (r, v) in [Rdi, Rsi, Rdx, Rcx].iter().zip(args) {
                        ctx.set(*r, *v);
                    }
                    cb(ctx)
                };
                let conn = call(&mut ctx, cb_create_connection, &[context, socket as u64, localhost]) as u64;
                ctx.write_bytes(base, b"ping");
                assert_eq!(call(&mut ctx, cb_connect, &[conn]), 0);
                assert_eq!(call(&mut ctx, cb_write, &[conn, base, 4, 0]), 4);
                assert_eq!(call(&mut ctx, cb_read, &[conn, base + 0x10, 16, 0]), 4);
                assert_eq!(ctx.read_bytes(base + 0x10, 4).unwrap(), b"pong");
                assert_eq!(call(&mut ctx, cb_read, &[conn, base + 0x10, 16, 0]), 0);
                assert_eq!(call(&mut ctx, cb_delete_connection, &[conn]), 0);
                assert!(crate::net::tests_close_socket(socket));
            })
        })
        .collect();
    clients.into_iter().for_each(|c| c.join().unwrap());
    assert_eq!(g.in_use(context), loaded);
    server.join().unwrap().unwrap();
    assert_eq!(g.call(cb_ssl_term, &[context]), 0);
}
