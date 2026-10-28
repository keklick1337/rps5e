// RustPS5Emu (RPS5E)
// Copyright (C) 2025-2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Bridge between the CPU trap handler and the HLE registry.
//!
//! Implements `ps5cpu::HleDispatcher` by looking up the guest stub address
//! in the stub table (addr → NID string), then dispatching through the HLE
//! registry (NID → callback). Guest register state is translated to/from
//! the HLE `CpuContext`.

#[cfg(windows)]
#[allow(unused_imports)]
use ps5hle::host::c as libc;
use ps5cpu::HleDispatcher;
use ps5hle::registry::HleExport;
use ps5hle::{CpuContext, HleRegistry};
use ps5vm::VirtualMemory;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

/// File descriptor for the async-signal-safe HLE trace file.
/// -1 = not opened. Opened lazily on first dispatch when PS5_HLE_TRACE=1.
static TRACE_FD: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(-2);

/// Write a u64 as hex into a buffer (async-signal-safe).
fn trace_hex(buf: &mut [u8], off: usize, val: u64) -> usize {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut n = 0usize;
    let mut started = false;
    for shift in (0..16).rev() {
        let nib = ((val >> (shift * 4)) & 0xf) as usize;
        if nib != 0 || started || shift == 0 {
            buf[off + n] = HEX[nib];
            n += 1;
            started = true;
        }
    }
    n
}

/// Append one HLE call record to the trace file. Fully async-signal-safe:
/// uses only libc::open/write, no allocation, no locks.
/// Format: "nid rdi rsi rdx rcx rax caller tid\n" (arguments as on entry,
/// tid = host thread id). `PS5_HLE_TRACE=<file>` writes to that file
/// (a relative path is relative to the current directory); `1` or an empty
/// value writes `hle_trace.txt` in the current directory.
fn trace_hle_call(nid: &str, args: [u64; 4], rax: u64, caller: u64) {
    let [rdi, rsi, rdx, rcx] = args;
    let fd = TRACE_FD.load(std::sync::atomic::Ordering::Relaxed);
    let fd = if fd == -2 {
        // First call — try to open the trace file.
        // Only enabled when PS5_HLE_TRACE env var is set.
        let Some(value) = std::env::var_os("PS5_HLE_TRACE") else {
            TRACE_FD.store(-1, std::sync::atomic::Ordering::Relaxed);
            return;
        };
        let value = value.to_string_lossy().into_owned();
        let path = if value.is_empty() || value == "1" { "hle_trace.txt".to_string() } else { value };
        let path = std::ffi::CString::new(path).unwrap_or_else(|_| c"hle_trace.txt".to_owned());
        let new_fd = unsafe {
            libc::open(
                path.as_ptr() as *const libc::c_char,
                libc::O_WRONLY | libc::O_CREAT | libc::O_TRUNC,
                0o644,
            )
        };
        TRACE_FD.store(new_fd, std::sync::atomic::Ordering::Relaxed);
        new_fd
    } else {
        fd
    };
    if fd < 0 {
        return;
    }
    // `PS5_HLE_TRACE_AFTER=<seconds>`: records start that long after the
    // first call (a late spin without gigabytes of start-up calls).
    static FROM: OnceLock<Option<std::time::Instant>> = OnceLock::new();
    let from = FROM.get_or_init(|| {
        let secs = std::env::var("PS5_HLE_TRACE_AFTER").ok()?.parse::<u64>().ok()?;
        Some(std::time::Instant::now() + std::time::Duration::from_secs(secs))
    });
    if from.is_some_and(|t| std::time::Instant::now() < t) {
        return;
    }
    let mut buf = [0u8; 192];
    let mut n = 0usize;
    // nid
    let nb = nid.as_bytes();
    let copy_len = nb.len().min(24);
    buf[n..n + copy_len].copy_from_slice(&nb[..copy_len]);
    n += copy_len;
    buf[n] = b' '; n += 1;
    // rdi
    n += trace_hex(&mut buf, n, rdi);
    buf[n] = b' '; n += 1;
    // rsi
    n += trace_hex(&mut buf, n, rsi);
    buf[n] = b' '; n += 1;
    // rdx
    n += trace_hex(&mut buf, n, rdx);
    buf[n] = b' '; n += 1;
    // rcx
    n += trace_hex(&mut buf, n, rcx);
    buf[n] = b' '; n += 1;
    // rax
    n += trace_hex(&mut buf, n, rax);
    buf[n] = b' '; n += 1;
    // caller (ret addr on guest stack)
    n += trace_hex(&mut buf, n, caller);
    buf[n] = b' '; n += 1;
    n += trace_hex(&mut buf, n, unsafe { libc::gettid() } as u64);
    buf[n] = b'\n'; n += 1;
    unsafe {
        libc::write(fd, buf.as_ptr() as *const _, n);
    }
}

/// Resolved import of one stub slot, cached on first call.
struct Slot {
    nid: &'static str,
    export: Option<HleExport>,
}

/// Maps guest stub addresses to NID strings and dispatches HLE calls.
///
/// The NID and callback of every stub slot are resolved once (first call)
/// and cached in a lock-free per-slot table, so a dispatch is an index
/// computation, one acquire load and the callback.
pub struct CoreDispatcher {
    addr_to_nid: Arc<std::sync::RwLock<HashMap<u64, String>>>,
    registry: Arc<HleRegistry>,
    vm: Arc<VirtualMemory>,
    slots: Box<[OnceLock<Slot>]>,
}

impl CoreDispatcher {
    pub fn new(
        addr_to_nid: Arc<std::sync::RwLock<HashMap<u64, String>>>,
        registry: Arc<HleRegistry>,
        vm: Arc<VirtualMemory>,
    ) -> Self {
        let slots = (0..crate::stubs::MAX_SLOTS).map(|_| OnceLock::new()).collect();
        Self { addr_to_nid, registry, vm, slots }
    }

    fn resolve(&self, slot_base: u64) -> Option<Slot> {
        let nid = self.addr_to_nid.read().unwrap().get(&slot_base)?.clone();
        let export = self.registry.lookup(&nid);
        match &export {
            Some(e) if !self.registry.is_implemented(&nid) => {
                log::info!("[HLE stub] first call: {} ({})", e.name, nid)
            }
            None => log::warn!("[dispatch] unknown NID {} at stub {:#x}", nid, slot_base),
            _ => {}
        }
        Some(Slot { nid: Box::leak(nid.into_boxed_str()), export })
    }

    fn slot(&self, slot_base: u64) -> Option<&Slot> {
        let i = crate::stubs::slot_index(slot_base)?;
        let cell = self.slots.get(i)?;
        if let Some(s) = cell.get() {
            return Some(s);
        }
        let s = self.resolve(slot_base)?;
        Some(cell.get_or_init(|| s))
    }
}

impl HleDispatcher for CoreDispatcher {
    fn dispatch_at(&self, stub_addr: u64, regs: &[u64; 16], _rsp: u64, _rip: u64) -> Option<u64> {
        // Stubs are 16-byte slots; a trap reports rip - 1 inside the slot.
        let slot = self.slot(stub_addr & !0xF)?;
        let Some(export) = slot.export else {
            return Some(0x8002_0002);
        };

        // The guest memory is identity-mapped, so the VM reference lets
        // callbacks read/write guest addresses directly.
        let mut ctx = CpuContext::new(&self.vm);
        ctx.regs = *regs;
        ctx.rip = stub_addr;

        let rax = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (export.callback)(&mut ctx))) {
            Ok(r) => r as u64,
            Err(payload) => {
                let msg = payload
                    .downcast_ref::<&str>()
                    .copied()
                    .or_else(|| payload.downcast_ref::<String>().map(|s| s.as_str()))
                    .unwrap_or("<non-string panic>");
                log::error!("[dispatch] PANIC in HLE callback nid={} ({}): {} — returning 0", slot.nid, export.name, msg);
                return Some(0);
            }
        };
        if TRACE_FD.load(std::sync::atomic::Ordering::Relaxed) != -1 {
            // caller = return address on the guest stack.
            let rsp = regs[4];
            let caller =
                if ps5vm::ranges::accessible(rsp, 8, false) { unsafe { std::ptr::read_volatile(rsp as *const u64) } } else { 0 };
            trace_hle_call(slot.nid, [regs[7], regs[6], regs[2], regs[1]], rax, caller);
        }
        Some(rax)
    }
}
