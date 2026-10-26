// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceZlib — zlib (RFC 1950) decompression service.
//!
//! The service decompresses a request completely when it is submitted, so
//! the request is already done when the caller waits for it. One request
//! can be outstanding at a time: its result is retrieved once, after which
//! the next request may be submitted. The scratch memory handed to
//! sceZlibInitialize is not needed.
//!
//! Streams with a bad header, bad data, a wrong Adler-32 checksum or
//! missing bytes complete with FATAL; output that does not fit in the
//! destination completes with NO_SPACE.

use std::sync::Mutex;

use miniz_oxide::inflate::core::{decompress, inflate_flags, DecompressorOxide};
use miniz_oxide::inflate::TINFLStatus;

use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};

const LIB: &str = "libSceZlib";

const fn err(code: u32) -> i32 {
    code as i32
}

const ERROR_NOT_FOUND: i32 = err(0x8112_0002);
const ERROR_BUSY: i32 = err(0x8112_000B);
const ERROR_FAULT: i32 = err(0x8112_000E);
const ERROR_INVALID: i32 = err(0x8112_0016);
const ERROR_NO_SPACE: i32 = err(0x8112_001C);
const ERROR_NOT_INITIALIZED: i32 = err(0x8112_0032);
const ERROR_ALREADY_INITIALIZED: i32 = err(0x8112_0033);
const ERROR_FATAL: i32 = err(0x8112_00FF);

/// Largest destination of one request.
const MAX_OUTPUT: u32 = 64 * 1024;
/// Required destination alignment.
const OUTPUT_ALIGNMENT: u64 = 2048;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Completion {
    id: u64,
    produced: u32,
    status: i32,
    /// Reported by sceZlibWaitForDone.
    waited: bool,
}

#[derive(Debug)]
struct Service {
    initialized: bool,
    next_id: u64,
    /// The request whose result has not been retrieved yet.
    pending: Option<Completion>,
}

impl Service {
    const fn new() -> Self {
        Self { initialized: false, next_id: 1, pending: None }
    }

    fn initialize(&mut self, buffer: u64, length: u64) -> i32 {
        if self.initialized {
            return ERROR_ALREADY_INITIALIZED;
        }
        // A scratch buffer comes with its size; no buffer, no size.
        if (buffer == 0) != (length == 0) {
            return ERROR_INVALID;
        }
        self.initialized = true;
        0
    }

    fn finalize(&mut self) -> i32 {
        if !self.initialized {
            return ERROR_NOT_INITIALIZED;
        }
        if self.pending.is_some() {
            return ERROR_BUSY;
        }
        self.initialized = false;
        0
    }

    /// Accept a request whose (status, produced bytes) are already known.
    fn submit(&mut self, produced: u32, status: i32) -> Result<u64, i32> {
        if self.pending.is_some() {
            return Err(ERROR_BUSY);
        }
        if self.next_id == u64::MAX {
            return Err(ERROR_BUSY);
        }
        let id = self.next_id;
        self.next_id += 1;
        self.pending = Some(Completion { id, produced, status, waited: false });
        Ok(id)
    }

    fn wait(&mut self) -> Result<u64, i32> {
        match self.pending.as_mut() {
            Some(c) if !c.waited => {
                c.waited = true;
                Ok(c.id)
            }
            _ => Err(ERROR_NOT_FOUND),
        }
    }

    fn result(&mut self, id: u64) -> Result<(u32, i32), i32> {
        match self.pending {
            Some(c) if c.id == id => {
                self.pending = None;
                Ok((c.produced, c.status))
            }
            _ => Err(ERROR_NOT_FOUND),
        }
    }
}

static SERVICE: Mutex<Service> = Mutex::new(Service::new());

fn service() -> std::sync::MutexGuard<'static, Service> {
    SERVICE.lock().unwrap_or_else(|p| p.into_inner())
}

/// Decompress a complete zlib stream into `out`: (bytes written, status).
fn inflate_zlib(input: &[u8], out: &mut [u8]) -> (usize, i32) {
    let mut decoder = Box::<DecompressorOxide>::default();
    let flags = inflate_flags::TINFL_FLAG_PARSE_ZLIB_HEADER
        | inflate_flags::TINFL_FLAG_COMPUTE_ADLER32
        | inflate_flags::TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF;
    let (status, _, produced) = decompress(&mut decoder, input, out, 0, flags);
    let status = match status {
        TINFLStatus::Done => 0,
        TINFLStatus::HasMoreOutput => ERROR_NO_SPACE,
        _ => ERROR_FATAL,
    };
    (produced, status)
}

/// sceZlibInitialize(const void* buffer, uint64_t length)
fn cb_initialize(ctx: &mut CpuContext) -> HleResult {
    service().initialize(ctx.rdi(), ctx.rsi()) as HleResult
}

/// sceZlibFinalize()
fn cb_finalize(_ctx: &mut CpuContext) -> HleResult {
    service().finalize() as HleResult
}

/// sceZlibInflate(const void* src, uint32_t srcLen, void* dst, uint32_t
/// dstLen, uint64_t* requestId)
fn cb_inflate(ctx: &mut CpuContext) -> HleResult {
    let (src, src_len, dst, dst_len, id_out) = (ctx.rdi(), ctx.rsi() as u32, ctx.rdx(), ctx.rcx() as u32, ctx.r8());
    let mut s = service();
    if !s.initialized {
        return ERROR_NOT_INITIALIZED as HleResult;
    }
    if src == 0 || src_len == 0 || dst == 0 || dst_len == 0 || id_out == 0 || dst_len > MAX_OUTPUT || dst % OUTPUT_ALIGNMENT != 0 {
        return ERROR_INVALID as HleResult;
    }
    if s.pending.is_some() {
        return ERROR_BUSY as HleResult;
    }
    let Some(input) = ctx.read_bytes(src, src_len as usize) else {
        return ERROR_FAULT as HleResult;
    };
    let mut out = vec![0u8; dst_len as usize];
    let (produced, status) = inflate_zlib(&input, &mut out);
    if !ctx.write_bytes(dst, &out[..produced]) {
        return ERROR_FAULT as HleResult;
    }
    let id = match s.submit(produced as u32, status) {
        Ok(id) => id,
        Err(e) => return e as HleResult,
    };
    if !ctx.write_u64(id_out, id) {
        // The request was not handed out.
        s.pending = None;
        return ERROR_FAULT as HleResult;
    }
    0
}

/// sceZlibWaitForDone(uint64_t* requestId, const uint32_t* timeout): the
/// outstanding request is already complete, so no timeout can run out.
fn cb_wait_for_done(ctx: &mut CpuContext) -> HleResult {
    let id_out = ctx.rdi();
    let mut s = service();
    if !s.initialized {
        return ERROR_NOT_INITIALIZED as HleResult;
    }
    if id_out == 0 {
        return ERROR_INVALID as HleResult;
    }
    let id = match s.wait() {
        Ok(id) => id,
        Err(e) => return e as HleResult,
    };
    if !ctx.write_u64(id_out, id) {
        if let Some(c) = s.pending.as_mut() {
            c.waited = false;
        }
        return ERROR_FAULT as HleResult;
    }
    0
}

/// sceZlibGetResult(uint64_t requestId, uint32_t* dstLen, int32_t* status):
/// the result of a request is retrieved once.
fn cb_get_result(ctx: &mut CpuContext) -> HleResult {
    let (id, len_out, status_out) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    let mut s = service();
    if !s.initialized {
        return ERROR_NOT_INITIALIZED as HleResult;
    }
    if len_out == 0 || status_out == 0 {
        return ERROR_INVALID as HleResult;
    }
    let pending = s.pending;
    let (produced, status) = match s.result(id) {
        Ok(r) => r,
        Err(e) => return e as HleResult,
    };
    if !ctx.write_u32(len_out, produced) || !ctx.write_u32(status_out, status as u32) {
        s.pending = pending;
        return ERROR_FAULT as HleResult;
    }
    0
}

const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("m1YErdIXCp4", "sceZlibInitialize", cb_initialize),
    ("6na+Sa-B83w", "sceZlibFinalize", cb_finalize),
    ("TLar1HULv1Q", "sceZlibInflate", cb_inflate),
    ("uB8VlDD4e0s", "sceZlibWaitForDone", cb_wait_for_done),
    ("2eDcGHC0YaM", "sceZlibGetResult", cb_get_result),
];

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, cb) in EXPORTS {
        reg.register_fn(nid, name, LIB, cb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CpuRegister::*;

    /// `TEXT` repeated 128 times, zlib-compressed.
    const COMPRESSED: [u8; 73] = [
        0x78, 0xda, 0xb, 0xa, 0x8, 0x36, 0x75, 0x55, 0xa8, 0xca, 0xc9, 0x4c, 0x52, 0x48, 0xce, 0x48, 0x4d, 0xce, 0x2e,
        0x2e, 0xcd, 0x55, 0x48, 0xcc, 0x4b, 0x51, 0x48, 0xca, 0x2f, 0xcd, 0x4b, 0x29, 0xd6, 0xe3, 0xa, 0x1a, 0x95, 0x1f,
        0x95, 0x1f, 0x95, 0x1f, 0x95, 0x1f, 0x95, 0x1f, 0x95, 0x1f, 0x95, 0x1f, 0x95, 0x1f, 0x95, 0x1f, 0x95, 0x1f, 0x95,
        0x1f, 0x95, 0x1f, 0x95, 0x1f, 0x95, 0x1f, 0x95, 0x1f, 0xf2, 0xf2, 0x0, 0xd2, 0x58, 0x74, 0xcc,
    ];
    const TEXT: &[u8] = b"RPS5E zlib checksum and bounds.\n";
    const EXPECTED_LEN: u32 = TEXT.len() as u32 * 128;

    #[test]
    fn nids_match_names() {
        for &(nid, name, _) in EXPORTS {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    #[test]
    fn inflate_status() {
        let mut out = vec![0u8; MAX_OUTPUT as usize];
        let (n, st) = inflate_zlib(&COMPRESSED, &mut out[..EXPECTED_LEN as usize]);
        assert_eq!((n as u32, st), (EXPECTED_LEN, 0));
        assert!(out[..n].chunks(TEXT.len()).all(|c| c == TEXT));
        let (n, st) = inflate_zlib(&COMPRESSED, &mut out[..EXPECTED_LEN as usize - 1]);
        assert_eq!(st, ERROR_NO_SPACE);
        assert!(n <= EXPECTED_LEN as usize - 1);
        let mut corrupt = COMPRESSED;
        *corrupt.last_mut().unwrap() ^= 1;
        assert_eq!(inflate_zlib(&corrupt, &mut out[..EXPECTED_LEN as usize]).1, ERROR_FATAL, "checksum");
        assert_eq!(inflate_zlib(&COMPRESSED[..COMPRESSED.len() - 1], &mut out).1, ERROR_FATAL, "truncated");
    }

    #[test]
    fn service_lifecycle() {
        let mut s = Service::new();
        assert_eq!(s.finalize(), ERROR_NOT_INITIALIZED);
        assert_eq!(s.initialize(0, 1), ERROR_INVALID);
        assert_eq!(s.initialize(0x1000, 0), ERROR_INVALID);
        assert_eq!(s.initialize(0, 0), 0);
        assert_eq!(s.initialize(0, 0), ERROR_ALREADY_INITIALIZED);
        let id = s.submit(10, 0).unwrap();
        assert_eq!(s.submit(10, 0), Err(ERROR_BUSY), "one outstanding request");
        assert_eq!(s.finalize(), ERROR_BUSY, "unread result");
        assert_eq!(s.wait(), Ok(id));
        assert_eq!(s.wait(), Err(ERROR_NOT_FOUND));
        assert_eq!(s.result(id + 1), Err(ERROR_NOT_FOUND));
        assert_eq!(s.result(id), Ok((10, 0)));
        assert_eq!(s.result(id), Err(ERROR_NOT_FOUND), "retrieved once");
        let next = s.submit(0, ERROR_FATAL).unwrap();
        assert!(next > id);
        assert_eq!(s.result(next), Ok((0, ERROR_FATAL)), "a result can be read without waiting");
        assert_eq!(s.finalize(), 0);
        assert_eq!(s.initialize(0x1000, 0x10000), 0);
    }

    #[test]
    fn guest_requests() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5F21_0000_0000;
        // [guard 2 KiB][output 64 KiB][guard 2 KiB][input, out params]
        vm.map(BASE, 0x20000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let output = BASE + 0x800;
        let input = BASE + 0x12000;
        let (id_addr, len_addr, status_addr) = (BASE + 0x13000, BASE + 0x13008, BASE + 0x13010);
        ctx.write_bytes(input, &COMPRESSED);
        let set = |ctx: &mut CpuContext, v: &[u64]| {
            for (r, x) in [Rdi, Rsi, Rdx, Rcx, R8].iter().zip(v) {
                ctx.set(*r, *x);
            }
        };
        let r = |v: i32| v as HleResult;

        // Every call needs an initialised service, checked first.
        set(&mut ctx, &[0, 0, 0, 0, 0]);
        assert_eq!(cb_finalize(&mut ctx), r(ERROR_NOT_INITIALIZED));
        assert_eq!(cb_inflate(&mut ctx), r(ERROR_NOT_INITIALIZED));
        assert_eq!(cb_wait_for_done(&mut ctx), r(ERROR_NOT_INITIALIZED));
        assert_eq!(cb_get_result(&mut ctx), r(ERROR_NOT_INITIALIZED));
        set(&mut ctx, &[0, 1]);
        assert_eq!(cb_initialize(&mut ctx), r(ERROR_INVALID));
        set(&mut ctx, &[0, 0]);
        assert_eq!(cb_initialize(&mut ctx), 0);
        assert_eq!(cb_initialize(&mut ctx), r(ERROR_ALREADY_INITIALIZED));

        // Destination alignment, size and pointers.
        let n = COMPRESSED.len() as u64;
        for bad in [
            [input, n, output + 1, EXPECTED_LEN as u64, id_addr],
            [input, n, output, MAX_OUTPUT as u64 + 1, id_addr],
            [0, n, output, EXPECTED_LEN as u64, id_addr],
            [input, n, output, EXPECTED_LEN as u64, 0],
        ] {
            set(&mut ctx, &bad);
            assert_eq!(cb_inflate(&mut ctx), r(ERROR_INVALID));
        }

        let previous = std::cell::Cell::new(0u64);
        let decode = |ctx: &mut CpuContext, src_len: u64, capacity: u32, expected: i32| -> u32 {
            ctx.write_bytes(BASE, &vec![0xA5; 0x800 + MAX_OUTPUT as usize + 0x800]);
            set(ctx, &[input, src_len, output, capacity as u64, id_addr]);
            assert_eq!(cb_inflate(ctx), 0);
            let id = ctx.read_u64(id_addr).unwrap();
            assert!(id > previous.get());
            previous.set(id);
            assert_eq!(cb_inflate(ctx), r(ERROR_BUSY), "one outstanding request");
            assert_eq!(cb_finalize(ctx), r(ERROR_BUSY), "unread result");
            set(ctx, &[BASE + 0x13020, 0]);
            assert_eq!(cb_wait_for_done(ctx), 0);
            assert_eq!(ctx.read_u64(BASE + 0x13020), Some(id));
            set(ctx, &[id + 1, len_addr, status_addr]);
            assert_eq!(cb_get_result(ctx), r(ERROR_NOT_FOUND));
            set(ctx, &[id, len_addr, status_addr]);
            assert_eq!(cb_get_result(ctx), 0);
            assert_eq!(cb_get_result(ctx), r(ERROR_NOT_FOUND), "retrieved once");
            let produced = ctx.read_u32(len_addr).unwrap();
            assert_eq!(ctx.read_u32(status_addr), Some(expected as u32));
            assert!(produced <= capacity);
            // Nothing outside the destination is written.
            assert_eq!(ctx.read_bytes(BASE, 0x800).unwrap(), vec![0xA5; 0x800]);
            let tail = 0x800 + MAX_OUTPUT as usize + 0x800 - (0x800 + capacity as usize);
            assert_eq!(ctx.read_bytes(output + capacity as u64, tail).unwrap(), vec![0xA5; tail]);
            produced
        };
        let produced = decode(&mut ctx, n, EXPECTED_LEN, 0);
        assert_eq!(produced, EXPECTED_LEN);
        let data = ctx.read_bytes(output, produced as usize).unwrap();
        assert!(data.chunks(TEXT.len()).all(|c| c == TEXT));
        decode(&mut ctx, n, EXPECTED_LEN - 1, ERROR_NO_SPACE);
        ctx.write_bytes(input + n - 1, &[COMPRESSED[COMPRESSED.len() - 1] ^ 1]);
        decode(&mut ctx, n, EXPECTED_LEN, ERROR_FATAL);
        ctx.write_bytes(input + n - 1, &[COMPRESSED[COMPRESSED.len() - 1]]);
        decode(&mut ctx, n - 1, MAX_OUTPUT, ERROR_FATAL);

        set(&mut ctx, &[]);
        assert_eq!(cb_finalize(&mut ctx), 0);
        set(&mut ctx, &[previous.get(), len_addr, status_addr]);
        assert_eq!(cb_get_result(&mut ctx), r(ERROR_NOT_INITIALIZED));
        set(&mut ctx, &[input, 0x10000]);
        assert_eq!(cb_initialize(&mut ctx), 0);
        decode(&mut ctx, n, MAX_OUTPUT, 0);
        assert_eq!(cb_finalize(&mut ctx), 0);
    }
}
