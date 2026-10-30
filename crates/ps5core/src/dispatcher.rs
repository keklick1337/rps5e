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

        // `RPS5E_HLE_PROFILE`: per-export call counts and time.
        let profile = ps5hle::registry::profile::enabled().then(std::time::Instant::now);
        let stuck = stuck::enabled();
        if stuck {
            stuck::enter(export.name, [regs[7], regs[6], regs[2], regs[1]], regs[4], regs[5], [regs[3], regs[12], regs[13], regs[14], regs[15]]);
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (export.callback)(&mut ctx)));
        if stuck {
            stuck::leave();
        }
        if let Some(start) = profile {
            ps5hle::registry::profile::record(export.nid, export.name, start.elapsed());
        }
        let rax = match result {
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

/// `RPS5E_HLE_STUCK=1` (debug aid): every guest thread's HLE call in
/// progress is recorded; [`stuck::report`] lists the calls running for 5 s
/// or more (a thread blocked in a wait, with its arguments and caller).
pub mod stuck {
    use std::cell::Cell;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Mutex, OnceLock};
    use std::time::{Duration, Instant};

    #[derive(Clone, Copy)]
    struct Call {
        name: &'static str,
        args: [u64; 4],
        caller: u64,
        /// Return addresses of the guest frames above (rbp chain).
        frames: [u64; 12],
        depth: usize,
        since: Instant,
        rsp: u64,
        /// rbx, r12-r15 at the call.
        saved: [u64; 5],
    }

    /// One guest thread's state; only that thread writes it (the report
    /// reads it), so its lock is uncontended on the call path.
    struct Slot {
        key: u64,
        thread: String,
        call: Mutex<Option<Call>>,
        /// HLE calls since the last busy report.
        calls: AtomicU64,
        sample: Mutex<String>,
    }

    fn slots() -> &'static Mutex<Vec<&'static Slot>> {
        static S: OnceLock<Mutex<Vec<&'static Slot>>> = OnceLock::new();
        S.get_or_init(Default::default)
    }

    thread_local! {
        static SLOT: Cell<Option<&'static Slot>> = const { Cell::new(None) };
    }

    fn slot() -> &'static Slot {
        SLOT.with(|c| match c.get() {
            Some(s) => s,
            None => {
                let me = std::thread::current();
                let key = {
                    use std::hash::{Hash, Hasher};
                    let mut h = std::collections::hash_map::DefaultHasher::new();
                    me.id().hash(&mut h);
                    h.finish()
                };
                // The slot lives as long as the process (threads are few).
                let s: &'static Slot = Box::leak(Box::new(Slot { key, thread: me.name().unwrap_or("").to_string(), call: Mutex::new(None), calls: AtomicU64::new(0), sample: Mutex::new(String::new()) }));
                slots().lock().unwrap_or_else(|e| e.into_inner()).push(s);
                c.set(Some(s));
                s
            }
        })
    }

    pub fn enabled() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var("RPS5E_HLE_STUCK").is_ok_and(|v| v == "1"))
    }

    /// The calling thread enters HLE call `name`.
    pub fn enter(name: &'static str, args: [u64; 4], rsp: u64, rbp: u64, saved: [u64; 5]) {
        let caller = if ps5vm::ranges::accessible(rsp, 8, false) {
            // SAFETY: readable guest stack slot (checked above).
            unsafe { std::ptr::read_volatile(rsp as *const u64) }
        } else {
            0
        };
        let mut frames = [0u64; 12];
        let mut depth = 0;
        let mut fp = rbp;
        while depth < frames.len() && fp & 7 == 0 && ps5vm::ranges::accessible(fp, 16, false) {
            // SAFETY: readable guest stack words (checked above).
            let (next, ret) = unsafe { (std::ptr::read_volatile(fp as *const u64), std::ptr::read_volatile((fp + 8) as *const u64)) };
            frames[depth] = ret;
            depth += 1;
            if next <= fp {
                break;
            }
            fp = next;
        }
        let s = slot();
        if s.calls.fetch_add(1, Ordering::Relaxed) % 64 == 0 {
            *s.sample.lock().unwrap_or_else(|e| e.into_inner()) = format!("last {name} args {args:x?} caller {caller:#x} frames {:x?}", &frames[..depth]);
        }
        *s.call.lock().unwrap_or_else(|e| e.into_inner()) = Some(Call { name, args, caller, frames, depth, since: Instant::now(), rsp, saved });
    }

    /// The threads that made the most HLE calls since the last call of this
    /// function (busy threads polling instead of waiting).
    pub fn busy_report() -> Vec<String> {
        let all = slots().lock().unwrap_or_else(|e| e.into_inner()).clone();
        let mut v: Vec<(u64, String)> = all
            .iter()
            .map(|s| (s.calls.swap(0, Ordering::Relaxed), format!("'{}' {}", s.thread, s.sample.lock().unwrap_or_else(|e| e.into_inner()))))
            .filter(|(n, _)| *n > 0)
            .collect();
        v.sort_by(|a, b| b.0.cmp(&a.0));
        v.into_iter().take(6).map(|(n, s)| format!("{n} calls: {s}")).collect()
    }

    /// The calling thread left its HLE call.
    pub fn leave() {
        *slot().call.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }

    /// One line per HLE call running for at least `min`.
    pub fn report(min: Duration) -> Vec<String> {
        let now = Instant::now();
        let all = slots().lock().unwrap_or_else(|e| e.into_inner()).clone();
        let mut v: Vec<(Duration, String)> = all
            .iter()
            .filter_map(|s| {
                let c = (*s.call.lock().unwrap_or_else(|e| e.into_inner()))?;
                (now - c.since >= min).then_some((s, c))
            })
            .map(|(s, c)| {
                let t = s.key;
                let mut line = format!("thread {t:016x} '{}' in {} for {:.1} s args {:x?} caller {:#x} frames {:x?}", s.thread, c.name, (now - c.since).as_secs_f64(), c.args, c.caller, &c.frames[..c.depth]);
                if c.name.contains("MutexLock") || c.name.contains("mutex_lock") {
                    if let Some((owner, name, depth, at)) = ps5hle::pthread::sync::describe_mutex(c.args[0]) {
                        line.push_str(&format!(" (mutex owner {owner:#x} '{name}' depth {depth}, locked at {:#x} <- {:#x})", at[0], at[1]));
                    }
                }
                if now - c.since >= Duration::from_secs(20) && std::env::var("RPS5E_HLE_STUCK_STACK").is_ok_and(|t| t == s.thread) {
                    // `RPS5E_HLE_STUCK_STACK=<thread name>`: the guest
                    // stack of that thread's long wait (callee-saved
                    // registers and locals of the frames above).
                    let words: Vec<String> = (0..160u64)
                        .map(|i| c.rsp + i * 8)
                        .take_while(|&a| ps5vm::ranges::accessible(a, 8, false))
                        // SAFETY: readable guest stack word (checked above).
                        .map(|a| format!("{:x}", unsafe { std::ptr::read_volatile(a as *const u64) }))
                        .collect();
                    line.push_str(&format!(" regs rbx/r12-r15 {:x?} rsp {:#x} stack [{}]", c.saved, c.rsp, words.join(" ")));
                }
                if c.name == "sceKernelWaitEventFlag" {
                    if let Some(d) = ps5hle::sync::event_flag::describe(c.args[0]) {
                        line.push_str(&format!(" (flag {d})"));
                    }
                }
                if c.name == "sceKernelWaitSema" {
                    if let Some(d) = ps5hle::sync::sema::describe(c.args[0] as u32) {
                        line.push_str(&format!(" (sema {d})"));
                    }
                }
                if c.name == "sceKernelWaitEqueue" {
                    if let Some(d) = ps5hle::sync::equeue::describe(c.args[0]) {
                        line.push_str(&format!(" (queue {d})"));
                    }
                }
                (now - c.since, line)
            })
            .collect();
        v.sort_by(|a, b| b.0.cmp(&a.0));
        v.into_iter().map(|(_, s)| s).collect()
    }
}
