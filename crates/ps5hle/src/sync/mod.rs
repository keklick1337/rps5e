// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libkernel synchronization objects: event queues, semaphores, event
//! flags and sync-on-address waits.
//!
//! Every object keeps its state on the host. Blocking calls block the
//! calling guest thread's host thread on a `Condvar` (the per-object or
//! per-subsystem mutex is released while waiting), so other guest threads
//! keep running. Timeouts are computed as `Instant` deadlines.
//!
//! Event queues are also the delivery channel for kernel-side event
//! sources (video-out flip/vblank, GPU end-of-pipe, AMPR completion, HR
//! timers); see the public API in [`equeue`].

pub mod equeue;
pub mod event_flag;
pub mod kqueue;
pub mod sema;
pub mod sync_on_address;

use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};
use std::sync::{Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

/// `RPS5E_REGISTRY_CACHE=1`: lookups of semaphores, event flags, event
/// queues and guest threads by handle hit a per-thread memo while the
/// registry is unchanged, instead of taking the registry's lock (default
/// off). With 16 threads each looking up its own object the shared lock
/// word cost 0.7–1.2 us per call against 20–70 ns alone.
pub(crate) fn registry_cache_enabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static MODE: AtomicU8 = AtomicU8::new(0);
    match REGISTRY_CACHE_FORCED.load(Ordering::Relaxed) {
        1 => return true,
        2 => return false,
        _ => {}
    }
    match MODE.load(Ordering::Relaxed) {
        0 => {
            let on = std::env::var("RPS5E_REGISTRY_CACHE").is_ok_and(|v| v.trim() == "1");
            MODE.store(if on { 1 } else { 2 }, Ordering::Relaxed);
            on
        }
        m => m == 1,
    }
}

static REGISTRY_CACHE_FORCED: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Force the registry memo on or off (tests and benchmarks; None: the
/// environment decides).
#[cfg(test)]
pub(crate) fn force_registry_cache(on: Option<bool>) {
    REGISTRY_CACHE_FORCED.store(on.map_or(0, |o| if o { 1 } else { 2 }), std::sync::atomic::Ordering::Relaxed);
}

/// Change counter of a registry: bumped (Release) after every insert or
/// removal, while the registry's write lock is held.
pub(crate) struct Generation(std::sync::atomic::AtomicU64);

impl Generation {
    pub(crate) const fn new() -> Generation {
        Generation(std::sync::atomic::AtomicU64::new(0))
    }

    pub(crate) fn bump(&self) {
        self.0.fetch_add(1, std::sync::atomic::Ordering::Release);
    }

    pub(crate) fn get(&self) -> u64 {
        self.0.load(std::sync::atomic::Ordering::Acquire)
    }
}

const LOOKUP_WAYS: usize = 4;

/// A thread's last registry lookups: (generation, key, object), replaced
/// round robin. An entry answers only at the generation it was read at:
/// any insert or removal since makes every thread look again.
pub(crate) struct LocalLookups<V> {
    slots: std::cell::RefCell<([Option<(u64, u64, std::sync::Arc<V>)>; LOOKUP_WAYS], usize)>,
}

impl<V> LocalLookups<V> {
    pub(crate) const fn new() -> LocalLookups<V> {
        LocalLookups { slots: std::cell::RefCell::new(([const { None }; LOOKUP_WAYS], 0)) }
    }

    /// The object of `key`: the memo at the registry's current generation,
    /// else `lookup` (under the registry's lock), remembered when found.
    pub(crate) fn get(&self, generation: &Generation, key: u64, lookup: impl FnOnce() -> Option<std::sync::Arc<V>>) -> Option<std::sync::Arc<V>> {
        if !registry_cache_enabled() {
            return lookup();
        }
        // Read before the lookup: a change after it moves the generation
        // past the one this entry is kept at.
        let g = generation.get();
        if let Ok(s) = self.slots.try_borrow() {
            if let Some(e) = s.0.iter().flatten().find(|e| e.0 == g && e.1 == key) {
                return Some(e.2.clone());
            }
        }
        // Entries of older generations go (their objects may be deleted).
        if let Ok(mut s) = self.slots.try_borrow_mut() {
            for e in s.0.iter_mut() {
                if e.as_ref().is_some_and(|e| e.0 != g) {
                    *e = None;
                }
            }
        }
        let v = lookup()?;
        if let Ok(mut s) = self.slots.try_borrow_mut() {
            let i = s.1;
            s.0[i] = Some((g, key, v.clone()));
            s.1 = (i + 1) % LOOKUP_WAYS;
        }
        Some(v)
    }
}

pub(crate) const OK: HleResult = 0;
pub(crate) const ERR_PERMISSION_DENIED: HleResult = 0x8002_0001u32 as i32 as i64;
pub(crate) const ERR_NOT_FOUND: HleResult = 0x8002_0002u32 as i32 as i64;
pub(crate) const ERR_INVALID_ARGUMENT: HleResult = 0x8002_0016u32 as i32 as i64;
pub(crate) const ERR_DELETED: HleResult = 0x8002_000Du32 as i32 as i64;
pub(crate) const ERR_BUSY: HleResult = 0x8002_0010u32 as i32 as i64;
pub(crate) const ERR_TRY_AGAIN: HleResult = 0x8002_0023u32 as i32 as i64;
pub(crate) const ERR_TIMED_OUT: HleResult = 0x8002_003Cu32 as i32 as i64;
pub(crate) const ERR_CANCELED: HleResult = 0x8002_0055u32 as i32 as i64;
pub(crate) const ERR_MEMORY_FAULT: HleResult = 0x8002_000Eu32 as i32 as i64;

/// Deadline for a relative timeout in microseconds.
pub(crate) fn usec_deadline(usec: u32) -> Instant {
    Instant::now() + Duration::from_micros(usec as u64)
}

/// Length of one kernel scheduler tick (hz = 1000).
pub(crate) const KERNEL_TICK_USEC: u32 = 1000;

/// Deadline of a kernel sleep with a relative timeout: the kernel converts
/// the timeout to scheduler ticks and sleeps for at least one, so a waiter
/// that is not satisfied at once gives up the CPU for a tick even when it
/// passed a zero timeout. Middleware polls this way (Wwise's event manager
/// waits on its event flag with a zero timeout in a loop); returning at
/// once turns that loop into a busy spin.
pub(crate) fn kernel_sleep_deadline(usec: u32) -> Instant {
    usec_deadline(usec.max(KERNEL_TICK_USEC))
}

/// Microseconds left until `deadline` (0 once it has passed).
pub(crate) fn remaining_usec(deadline: Instant) -> u32 {
    let left = deadline.saturating_duration_since(Instant::now());
    left.as_micros().min(u32::MAX as u128) as u32
}

/// Outcome of one bounded condvar wait.
pub(crate) enum Waited<'a, T> {
    Woken(MutexGuard<'a, T>),
    TimedOut(MutexGuard<'a, T>),
}

/// Wait on `cv` (whose mutex is `m`) once, until notified, `deadline`
/// (None = forever) or a bounded slice — `slice` shortens it so callers can
/// re-evaluate external state. Pending guest signals are delivered with
/// the lock released (see `crate::signals`); `Woken` may be spurious.
pub(crate) fn wait_until<'a, T>(
    m: &'a Mutex<T>,
    cv: &Condvar,
    guard: MutexGuard<'a, T>,
    deadline: Option<Instant>,
    slice: Option<Duration>,
) -> Waited<'a, T> {
    match crate::signals::wait_cv_slice(m, cv, guard, deadline, slice) {
        (g, true) => Waited::TimedOut(g),
        (g, false) => Waited::Woken(g),
    }
}

/// Read a guest NUL-terminated name of at most `max` bytes. Reads byte by
/// byte so a name near the end of a mapping does not fault.
pub(crate) fn read_name(ctx: &CpuContext, addr: u64, max: usize) -> Option<String> {
    let mut bytes = Vec::new();
    for i in 0..max as u64 {
        let mut b = [0u8; 1];
        ctx.vm.read(addr + i, &mut b).ok()?;
        if b[0] == 0 {
            break;
        }
        bytes.push(b[0]);
    }
    Some(String::from_utf8_lossy(&bytes).into_owned())
}

/// One export: (NID, name, library, callback).
pub(crate) type Export = (&'static str, &'static str, &'static str, HleCallback);

fn all_exports() -> impl Iterator<Item = &'static Export> {
    equeue::EXPORTS
        .iter()
        .chain(sema::EXPORTS)
        .chain(event_flag::EXPORTS)
        .chain(sync_on_address::EXPORTS)
        .chain(kqueue::EXPORTS)
}

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, lib, cb) in all_exports() {
        reg.register_fn(nid, name, lib, cb);
    }
}

#[cfg(test)]
pub(crate) mod test_util {
    use base64::Engine as _;
    use sha1::{Digest, Sha1};

    /// NID = base64(reverse(first 8 bytes of SHA1(name || salt))) with the
    /// '+' / '-' alphabet and no padding.
    pub fn nid(name: &str) -> String {
        const SALT: [u8; 16] = [
            0x51, 0x8D, 0x64, 0xA6, 0x35, 0xDE, 0xD8, 0xC1, 0xE6, 0xB0, 0x39, 0xB1, 0xC3, 0xE5,
            0x52, 0x30,
        ];
        let mut h = Sha1::new();
        h.update(name.as_bytes());
        h.update(SALT);
        let d = h.finalize();
        let mut b = [0u8; 8];
        b.copy_from_slice(&d[..8]);
        b.reverse();
        base64::engine::general_purpose::STANDARD_NO_PAD.encode(b).replace('/', "-")
    }

    #[test]
    fn registered_nids_match_names() {
        let mut n = 0;
        for &(id, name, _, _) in super::all_exports() {
            assert_eq!(nid(name), id, "NID mismatch for {name}");
            n += 1;
        }
        assert!(n > 40, "only {n} exports registered");
    }

    /// A small guest-memory arena for callback-level tests.
    pub fn vm_arena(base: u64) -> &'static ps5vm::VirtualMemory {
        let vm: &'static ps5vm::VirtualMemory = Box::leak(Box::new(ps5vm::VirtualMemory::new()));
        vm.map(base, 0x10000, ps5vm::MemProtect::RW).expect("map test arena");
        vm
    }
}

#[cfg(test)]
mod registry_memo_tests {
    use super::*;

    /// One thread's answers for objects another thread creates and
    /// deletes, with the per-thread memo off and on: the same (a deleted
    /// object is not found again, a new one is).
    fn answers() -> Vec<i64> {
        let mut out = Vec::new();
        let h = sema::create("memo".into(), false, 0, 10);
        out.push(sema::signal(h, 1));
        out.push(sema::poll(h, 1));
        std::thread::spawn(move || assert!(sema::delete(h))).join().unwrap();
        out.push(sema::signal(h, 1));
        let h2 = std::thread::spawn(|| sema::create("memo2".into(), false, 1, 10)).join().unwrap();
        out.push(sema::poll(h2, 1));
        let q = equeue::create();
        out.push(equeue::is_valid_equeue(q) as i64);
        std::thread::spawn(move || assert!(equeue::delete(q))).join().unwrap();
        out.push(equeue::is_valid_equeue(q) as i64);
        let f = event_flag::create("memo".into(), event_flag::ATTR_MULTI, 1);
        out.push(event_flag::poll(f, 1, event_flag::WAITMODE_OR).0);
        std::thread::spawn(move || assert!(event_flag::delete(f))).join().unwrap();
        out.push(event_flag::poll(f, 1, event_flag::WAITMODE_OR).0);
        let t = std::thread::spawn(|| crate::pthread::tests::fake_thread("memo", 1)).join().unwrap().0;
        out.push(crate::pthread::thread_exists(t) as i64);
        std::thread::spawn(move || crate::pthread::tests::forget_fake_thread(t)).join().unwrap();
        out.push(crate::pthread::thread_exists(t) as i64);
        out
    }

    #[test]
    fn registry_memo_follows_other_threads_changes() {
        force_registry_cache(Some(false));
        let off = answers();
        force_registry_cache(Some(true));
        let on = answers();
        // Twice: the second round meets memo entries of the first.
        let again = answers();
        force_registry_cache(None);
        assert_eq!(off, on);
        assert_eq!(on, again);
        assert_eq!(off[0], OK);
        assert_ne!(off[2], OK, "a deleted semaphore is not found");
        assert_eq!((off[4], off[5]), (1, 0), "a deleted queue is not valid");
        assert_eq!((off[8], off[9]), (1, 0), "a forgotten thread does not exist");
    }
}
