// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Kernel event queues (`SceKernelEqueue`, a kqueue wrapper).
//!
//! A queue holds a list of registered events keyed by `(ident, filter)`.
//! Each registration carries its current `SceKernelEvent`, a `triggered`
//! flag, a FIFO of further pending occurrences and optional filter hooks:
//! `trigger` (how a trigger updates the event), `reset` (what delivery does
//! to it) and `on_delete` (run once when the registration or the queue
//! goes away). Delivery walks the list: a triggered event is copied out;
//! `EV_ONESHOT` events are then removed, events with a reset hook or
//! `EV_CLEAR` are reset, and the next pending occurrence (if any) becomes
//! current. A level event (no `EV_CLEAR`) stays triggered and is delivered
//! again by the next wait, never twice in one batch. A reported event moves
//! behind the ones still waiting, so level events take turns.
//!
//! Timers live in the same list: a registration with a deadline is
//! triggered when the deadline passes (periodic `EVFILT_TIMER` events count
//! elapsed periods in `data`), and waits sleep no longer than the nearest
//! deadline, so no timer thread is needed.
//!
//! Queue handles are never reused; a deleted handle stays invalid (EBADF).
//! Every queue has its own lock, so independent queues never contend.
//!
//! Kernel-side event sources use the public API below:
//!   - [`trigger_display_event`]: video-out flip / vblank / output-mode.
//!   - [`register_event`] + [`trigger_registered_events_by_filter`]: GPU
//!     end-of-pipe interrupts (one occurrence per trigger).
//!   - [`trigger_ampr_event`]: AMPR completion.
//!   - [`add_read_readiness_provider`]: descriptor readability for
//!     `sceKernelAddReadEvent`; kernel sockets are known directly
//!     (`crate::net::socket_readiness`, read and write). Descriptor
//!     registrations are level events polled while a wait sleeps (1 ms
//!     slices) and dropped when their descriptor closes.
//!
//! `SceKernelEvent` layout (0x20 bytes):
//!   +0x00 u64 ident, +0x08 i16 filter, +0x0A u16 flags, +0x0C u32 fflags,
//!   +0x10 u64 data, +0x18 u64 udata.

use super::{Export, ERR_INVALID_ARGUMENT, ERR_MEMORY_FAULT, ERR_NOT_FOUND, ERR_TIMED_OUT, OK};
use crate::context::CpuContext;
use crate::registry::HleResult;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, RwLock};
use std::time::{Duration, Instant};

/// EBADF: not a live event queue.
pub(crate) const ERR_BAD_QUEUE: HleResult = 0x8002_0009u32 as i32 as i64;

/// Size of `SceKernelEvent`.
pub const KERNEL_EVENT_SIZE: u64 = 0x20;

/// Event filters.
pub const FILTER_READ: i16 = -1;
pub const FILTER_WRITE: i16 = -2;
pub const FILTER_FILE: i16 = -4;
pub const FILTER_TIMER: i16 = -7;
pub const FILTER_USER: i16 = -11;
pub const FILTER_VIDEO_OUT: i16 = -13;
pub const FILTER_GRAPHICS: i16 = -14;
pub const FILTER_HR_TIMER: i16 = -15;
pub const FILTER_AMPR: i16 = -25;

/// Event flags.
pub const EV_ADD: u16 = 0x0001;
pub const EV_ONESHOT: u16 = 0x0010;
pub const EV_CLEAR: u16 = 0x0020;
/// Disable the registration after each delivery (kqueue).
pub const EV_DISPATCH: u16 = 0x0080;
pub const EV_ERROR: u16 = 0x4000;
pub const EV_EOF: u16 = 0x8000;

/// Video-out event identifiers (ident of [`FILTER_VIDEO_OUT`] events).
pub const VIDEO_OUT_EVENT_FLIP: u64 = 0x6;
pub const VIDEO_OUT_EVENT_OUTPUT_MODE: u64 = 0x8;
pub const VIDEO_OUT_EVENT_VBLANK: u64 = 0x40;
pub const VIDEO_OUT_EVENT_PRE_VBLANK_START: u64 = 0x2;
/// VRR active-status change (event id 16).
pub const VIDEO_OUT_EVENT_VRR_STATUS: u64 = 0x10;

/// One `SceKernelEvent`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KernelEvent {
    pub ident: u64,
    pub filter: i16,
    pub flags: u16,
    pub fflags: u32,
    pub data: u64,
    pub udata: u64,
}

impl KernelEvent {
    pub fn to_bytes(&self) -> [u8; KERNEL_EVENT_SIZE as usize] {
        let mut b = [0u8; KERNEL_EVENT_SIZE as usize];
        b[0x00..0x08].copy_from_slice(&self.ident.to_le_bytes());
        b[0x08..0x0A].copy_from_slice(&self.filter.to_le_bytes());
        b[0x0A..0x0C].copy_from_slice(&self.flags.to_le_bytes());
        b[0x0C..0x10].copy_from_slice(&self.fflags.to_le_bytes());
        b[0x10..0x18].copy_from_slice(&self.data.to_le_bytes());
        b[0x18..0x20].copy_from_slice(&self.udata.to_le_bytes());
        b
    }
}

/// Run when a registration is deleted or its queue closes: (queue, event).
pub type DeleteHook = Arc<dyn Fn(u64, &mut EqueueEvent) + Send + Sync>;

/// Filter hooks of a registration.
#[derive(Clone, Default)]
pub struct Filter {
    /// How a trigger with `data` updates the event (default: mark triggered).
    pub trigger: Option<fn(&mut EqueueEvent, u64)>,
    /// What delivery does to the event (default: clear it if `EV_CLEAR`).
    pub reset: Option<fn(&mut EqueueEvent)>,
    pub on_delete: Option<DeleteHook>,
    /// Low-water mark of a read registration.
    pub low_water: u64,
    /// Owner of a vnode registration (the watch that feeds it).
    pub watch: u64,
    /// Notes a vnode registration reports (NOTE_WRITE, ...).
    pub notes: u32,
}

/// One registration on a queue.
#[derive(Clone, Default)]
pub struct EqueueEvent {
    pub triggered: bool,
    /// Monotonic deadline in [`now_ns`] units (0: none).
    pub deadline_ns: u64,
    /// Period of a repeating timer (0: none / fires once per delivery).
    pub interval_ns: u64,
    pub event: KernelEvent,
    pub filter: Filter,
    pub pending: VecDeque<KernelEvent>,
    /// EV_DISABLE (or delivered with EV_DISPATCH): kept, never delivered,
    /// triggers ignored.
    pub disabled: bool,
}

/// Monotonic nanoseconds; never below 1 ms, so a deadline of 1 is long past.
pub fn now_ns() -> u64 {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_nanos() as u64 + 1_000_000
}

struct QueueState {
    events: Vec<EqueueEvent>,
    closed: bool,
    sleepers: u32,
}

/// A live event queue.
pub struct Equeue {
    handle: u64,
    name: String,
    state: Mutex<QueueState>,
    cv: Condvar,
}

fn lock_state(q: &Equeue) -> MutexGuard<'_, QueueState> {
    q.state.lock().unwrap_or_else(|e| e.into_inner())
}

static QUEUES: RwLock<Option<HashMap<u64, Arc<Equeue>>>> = RwLock::new(None);
static NEXT_HANDLE: AtomicU64 = AtomicU64::new(1);

/// The live queue behind `handle`; holding the Arc keeps it alive.
pub fn pin(handle: u64) -> Option<Arc<Equeue>> {
    if handle == 0 {
        return None;
    }
    let lookup = || QUEUES.read().unwrap_or_else(|e| e.into_inner()).as_ref()?.get(&handle).cloned();
    QUEUE_LOOKUPS.try_with(|c| c.get(&QUEUES_GENERATION, handle, lookup)).unwrap_or_else(|_| lookup())
}

static QUEUES_GENERATION: super::Generation = super::Generation::new();

/// Diagnostics: the queue's name and registrations as
/// `(ident, filter, triggered, disabled, pending)`.
pub fn describe(handle: u64) -> Option<String> {
    let q = QUEUES.read().unwrap_or_else(|e| e.into_inner()).as_ref()?.get(&handle).cloned()?;
    let s = lock_state(&q);
    let ev: Vec<String> = s.events.iter().map(|e| format!("({:#x},{},{},{},{})", e.event.ident, e.event.filter, e.triggered, e.disabled, e.pending.len())).collect();
    Some(format!("'{}' [{}]", q.name, ev.join(" ")))
}

thread_local! {
    static QUEUE_LOOKUPS: super::LocalLookups<Equeue> = const { super::LocalLookups::new() };
}

fn all_queues() -> Vec<Arc<Equeue>> {
    QUEUES.read().unwrap_or_else(|e| e.into_inner()).as_ref().map_or_else(Vec::new, |m| m.values().cloned().collect())
}

impl Equeue {
    fn notify(&self, s: &QueueState) {
        if s.sleepers > 0 {
            self.cv.notify_all();
        }
    }

    fn find(s: &mut QueueState, ident: u64, filter: i16) -> Option<usize> {
        s.events.iter().position(|e| e.event.ident == ident && e.event.filter == filter)
    }

    /// Register `event`; an existing `(ident, filter)` keeps its state and
    /// hooks and only takes the new deadline, interval and user data.
    pub fn add_event(&self, event: EqueueEvent) -> HleResult {
        let mut s = lock_state(self);
        if s.closed {
            return ERR_BAD_QUEUE;
        }
        match Self::find(&mut s, event.event.ident, event.event.filter) {
            Some(i) => {
                trigger_expired_timers(&mut s.events, now_ns());
                let e = &mut s.events[i];
                e.deadline_ns = event.deadline_ns;
                e.interval_ns = event.interval_ns;
                e.event.udata = event.event.udata;
                for p in e.pending.iter_mut() {
                    p.udata = event.event.udata;
                }
            }
            None => s.events.push(event),
        }
        self.notify(&s);
        OK
    }

    pub fn trigger_event(&self, ident: u64, filter: i16, data: u64) -> HleResult {
        let mut s = lock_state(self);
        if s.closed {
            return ERR_BAD_QUEUE;
        }
        let Some(i) = Self::find(&mut s, ident, filter) else { return ERR_NOT_FOUND };
        let e = &mut s.events[i];
        if !e.disabled {
            match e.filter.trigger {
                Some(f) => f(e, data),
                None => e.triggered = true,
            }
        }
        self.notify(&s);
        OK
    }

    /// True when `(ident, filter)` is registered.
    pub fn has_event(&self, ident: u64, filter: i16) -> bool {
        let mut s = lock_state(self);
        Self::find(&mut s, ident, filter).is_some()
    }

    /// Change the `(ident, filter)` registration in place and wake waiters.
    pub fn edit_event(&self, ident: u64, filter: i16, f: impl FnOnce(&mut EqueueEvent)) -> HleResult {
        let mut s = lock_state(self);
        if s.closed {
            return ERR_BAD_QUEUE;
        }
        let Some(i) = Self::find(&mut s, ident, filter) else { return ERR_NOT_FOUND };
        f(&mut s.events[i]);
        self.notify(&s);
        OK
    }

    /// Drop the read, write and vnode registrations of descriptor `ident`,
    /// running their delete hooks.
    fn remove_descriptor_events(&self, ident: u64) {
        let mut s = lock_state(self);
        let mut gone = Vec::new();
        let mut i = 0;
        while i < s.events.len() {
            let e = &s.events[i];
            if e.event.ident == ident && (is_descriptor_filter(e.event.filter) || e.event.filter == FILTER_FILE) {
                gone.push(s.events.remove(i));
            } else {
                i += 1;
            }
        }
        if gone.is_empty() {
            return;
        }
        for e in gone.iter_mut() {
            if let Some(hook) = e.filter.on_delete.clone() {
                hook(self.handle, e);
            }
        }
        self.notify(&s);
    }

    pub fn delete_event(&self, ident: u64, filter: i16) -> HleResult {
        let mut s = lock_state(self);
        if s.closed {
            return ERR_BAD_QUEUE;
        }
        let Some(i) = Self::find(&mut s, ident, filter) else { return ERR_NOT_FOUND };
        let mut e = s.events.remove(i);
        if let Some(hook) = e.filter.on_delete.clone() {
            hook(self.handle, &mut e);
        }
        OK
    }

    /// Close: run every delete hook once and wake all waiters.
    fn close(&self) {
        let mut s = lock_state(self);
        if s.closed {
            return;
        }
        s.closed = true;
        let mut events = std::mem::take(&mut s.events);
        for e in events.iter_mut() {
            if let Some(hook) = e.filter.on_delete.clone() {
                hook(self.handle, e);
            }
        }
        drop(events);
        self.cv.notify_all();
    }

    /// Copy out up to `cap` triggered events (see the module docs).
    fn collect(&self, s: &mut QueueState, cap: usize) -> Result<Vec<KernelEvent>, HleResult> {
        if s.closed {
            return Err(ERR_BAD_QUEUE);
        }
        trigger_expired_timers(&mut s.events, now_ns());
        refresh_read_events(&mut s.events);
        let mut out = Vec::new();
        // Every registration a wait reports moves behind the ones still
        // waiting, in the order reported, as the kernel requeues a knote at
        // the tail of its active list: two level events take turns.
        let events = std::mem::take(&mut s.events);
        let mut waiting = Vec::with_capacity(events.len());
        let mut reported = Vec::new();
        for mut e in events {
            if out.len() >= cap || !e.triggered || e.disabled {
                waiting.push(e);
                continue;
            }
            let mut erase = false;
            while e.triggered && !e.disabled {
                out.push(e.event);
                if e.event.flags & EV_ONESHOT != 0 {
                    erase = true;
                    break;
                }
                if e.event.flags & EV_DISPATCH != 0 {
                    e.disabled = true;
                    e.triggered = false;
                    e.event.fflags = 0;
                    e.event.data = 0;
                    break;
                }
                match e.filter.reset {
                    Some(f) => f(&mut e),
                    None if e.event.flags & EV_CLEAR != 0 => {
                        e.triggered = false;
                        e.event.fflags = 0;
                        e.event.data = 0;
                    }
                    None => {}
                }
                match e.pending.pop_front() {
                    Some(next) => {
                        e.event = next;
                        e.triggered = true;
                    }
                    // A persistent level is delivered again by the next wait.
                    None => break,
                }
                if out.len() >= cap {
                    break;
                }
            }
            if erase {
                if let Some(hook) = e.filter.on_delete.clone() {
                    hook(self.handle, &mut e);
                }
            } else {
                reported.push(e);
            }
        }
        waiting.append(&mut reported);
        s.events = waiting;
        Ok(out)
    }

    /// Wait for events: `timeout_us` None = forever.
    fn wait_for_events(&self, cap: usize, timeout_us: Option<u32>) -> Result<Vec<KernelEvent>, HleResult> {
        self.wait_until(cap, timeout_us.map(|us| Instant::now() + Duration::from_micros(us as u64)))
    }

    /// Wait for events until `deadline` (None = forever); a deadline already
    /// past collects once.
    pub(crate) fn wait_until(&self, cap: usize, deadline: Option<Instant>) -> Result<Vec<KernelEvent>, HleResult> {
        let mut s = lock_state(self);
        loop {
            let out = self.collect(&mut s, cap)?;
            if !out.is_empty() {
                return Ok(out);
            }
            if deadline.is_some_and(|d| Instant::now() >= d) {
                return Ok(out);
            }
            let mut timer_wait = next_timer_wait(&s.events, now_ns());
            if s.events.iter().any(|e| is_descriptor_filter(e.event.filter) && !e.disabled) {
                // Descriptor readiness is polled: no one signals it.
                timer_wait = Some(timer_wait.map_or(DESCRIPTOR_POLL_SLICE, |t| t.min(DESCRIPTOR_POLL_SLICE)));
            }
            s.sleepers += 1;
            s = crate::signals::wait_cv_slice(&self.state, &self.cv, s, deadline, timer_wait).0;
            s.sleepers -= 1;
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn handle(&self) -> u64 {
        self.handle
    }
}

fn trigger_expired_timers(events: &mut [EqueueEvent], now: u64) {
    for e in events.iter_mut() {
        if !e.disabled && e.deadline_ns != 0 && e.deadline_ns <= now {
            if e.event.filter == FILTER_TIMER {
                let count = if e.interval_ns == 0 {
                    (!e.triggered) as u64
                } else {
                    1 + (now - e.deadline_ns) / e.interval_ns
                };
                e.event.data = e.event.data.wrapping_add(count);
                e.deadline_ns = e.deadline_ns.saturating_add(count.saturating_mul(e.interval_ns));
            }
            e.triggered = true;
        }
    }
}

/// Time until the nearest deadline of an untriggered registration.
fn next_timer_wait(events: &[EqueueEvent], now: u64) -> Option<Duration> {
    let nearest = events.iter().filter(|e| !e.disabled && !e.triggered && e.deadline_ns != 0).map(|e| e.deadline_ns).min()?;
    let remaining = nearest.saturating_sub(now);
    Some(Duration::from_micros(remaining.div_ceil(1000).max(1)))
}

// ---------------------------------------------------------------------------
// Filter hooks
// ---------------------------------------------------------------------------

/// User events: the trigger argument becomes both data and user data.
fn user_trigger(e: &mut EqueueEvent, data: u64) {
    e.triggered = true;
    e.event.data = data;
    e.event.udata = data;
}

/// User/AMPR reset: only edge (`EV_CLEAR`) events clear on delivery.
fn user_reset(e: &mut EqueueEvent) {
    if e.event.flags & EV_CLEAR != 0 {
        e.triggered = false;
        e.event.fflags = 0;
        e.event.data = 0;
    }
}

/// AMPR completions queue up: one occurrence per trigger.
fn ampr_trigger(e: &mut EqueueEvent, data: u64) {
    let next = KernelEvent { data, ..e.event };
    if e.triggered {
        e.pending.push_back(next);
    } else {
        e.event = next;
        e.triggered = true;
    }
}

/// GPU interrupts: one occurrence per trigger, counted in fflags.
fn interrupt_trigger(e: &mut EqueueEvent, data: u64) {
    let next = KernelEvent { fflags: e.event.fflags.wrapping_add(1), data, ..e.event };
    if e.triggered {
        e.pending.push_back(next);
    } else {
        e.event = next;
        e.triggered = true;
    }
}

fn interrupt_reset(e: &mut EqueueEvent) {
    e.triggered = false;
    e.event.fflags = 0;
    e.event.data = 0;
}

/// Hooks for registrations made through [`register_event`].
fn filter_for(filter: i16, low_water: u64) -> Filter {
    match filter {
        FILTER_USER => Filter { trigger: Some(user_trigger), reset: Some(user_reset), ..Default::default() },
        FILTER_AMPR => Filter { trigger: Some(ampr_trigger), reset: Some(user_reset), ..Default::default() },
        FILTER_GRAPHICS => Filter { trigger: Some(interrupt_trigger), reset: Some(interrupt_reset), ..Default::default() },
        _ => Filter { low_water, ..Default::default() },
    }
}

// ---------------------------------------------------------------------------
// Read-event readiness
// ---------------------------------------------------------------------------

/// Readiness of a descriptor for `FILTER_READ`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadReadiness {
    pub ready: bool,
    pub available: u64,
    /// Event flags to report (e.g. [`EV_EOF`]).
    pub flags: u16,
}

/// Returns `None` when the provider does not know `fd`.
pub type ReadReadinessProvider = fn(fd: i32, low_water: u64) -> Option<ReadReadiness>;

static READ_PROVIDERS: RwLock<Vec<ReadReadinessProvider>> = RwLock::new(Vec::new());

/// Register a source of descriptor readiness (files, sockets, ...).
pub fn add_read_readiness_provider(p: ReadReadinessProvider) {
    READ_PROVIDERS.write().unwrap_or_else(|e| e.into_inner()).push(p);
}

/// Readiness of `fd` for `FILTER_READ` or `FILTER_WRITE`: the registered
/// providers (read only), then the kernel sockets.
fn descriptor_readiness(fd: i32, filter: i16, low_water: u64) -> Option<ReadReadiness> {
    if filter == FILTER_READ {
        let providers = READ_PROVIDERS.read().unwrap_or_else(|e| e.into_inner());
        if let Some(r) = providers.iter().find_map(|p| p(fd, low_water)) {
            return Some(r);
        }
        // The read end of a pipe: ready with data or once every writer is
        // gone; the low-water mark does not apply.
        if let Some((available, eof)) = crate::filesystem::pipe::read_state(fd) {
            return Some(ReadReadiness { ready: available > 0 || eof, available, flags: if eof { EV_EOF } else { 0 } });
        }
    } else if let Some((free, eof, writable)) = crate::filesystem::pipe::write_state(fd) {
        // The write end of a pipe: ready while the host takes a write, with
        // the free buffer space; EV_EOF with no space once the reader is gone.
        return Some(ReadReadiness { ready: writable || eof, available: free, flags: if eof { EV_EOF } else { 0 } });
    }
    let mut r = crate::net::socket_readiness(fd, filter == FILTER_WRITE)?;
    // A socket registration waits for its low-water mark (or EOF): bytes
    // to read, or free send space for a write registration.
    if r.ready && r.flags & EV_EOF == 0 && r.available < low_water && !crate::net::is_listening_socket(fd) {
        r.ready = false;
    }
    Some(r)
}

fn is_descriptor_filter(filter: i16) -> bool {
    filter == FILTER_READ || filter == FILTER_WRITE
}

/// Longest sleep of a wait while descriptor registrations are polled.
const DESCRIPTOR_POLL_SLICE: Duration = Duration::from_millis(1);

/// Drop the read and write registrations of a closed descriptor from every
/// queue.
pub fn descriptor_closed(fd: i32) {
    for q in all_queues() {
        q.remove_descriptor_events(fd as u32 as u64);
    }
}

/// Level-triggered read and write registrations follow the descriptor's
/// readiness; a descriptor that disappeared drops its registration.
fn refresh_read_events(events: &mut Vec<EqueueEvent>) {
    if !events.iter().any(|e| is_descriptor_filter(e.event.filter)) {
        return;
    }
    events.retain_mut(|e| {
        if !is_descriptor_filter(e.event.filter) || e.disabled {
            return true;
        }
        match descriptor_readiness(e.event.ident as u32 as i32, e.event.filter, e.filter.low_water.max(1)) {
            None => false,
            Some(r) => {
                e.triggered = r.ready;
                e.event.data = if r.ready { r.available } else { 0 };
                // The registration's own flags stay; EV_EOF follows the descriptor.
                e.event.flags = (e.event.flags & !EV_EOF) | (r.flags & EV_EOF);
                true
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Public kernel-side API
// ---------------------------------------------------------------------------

/// Create a queue; handles are never recycled.
pub fn create_named(name: &str) -> u64 {
    let handle = NEXT_HANDLE.fetch_add(1, Ordering::Relaxed);
    let q = Arc::new(Equeue {
        handle,
        name: name.to_string(),
        state: Mutex::new(QueueState { events: Vec::new(), closed: false, sleepers: 0 }),
        cv: Condvar::new(),
    });
    {
        let mut m = QUEUES.write().unwrap_or_else(|e| e.into_inner());
        m.get_or_insert_with(HashMap::new).insert(handle, q);
        QUEUES_GENERATION.bump();
    }
    handle
}

#[cfg(test)]
pub(crate) fn create() -> u64 {
    create_named("")
}

/// Remove the queue from the registry, then close it (delete hooks run
/// after the handle is already invalid).
pub(crate) fn delete(handle: u64) -> bool {
    let q = {
        let mut m = QUEUES.write().unwrap_or_else(|e| e.into_inner());
        let q = m.as_mut().and_then(|m| m.remove(&handle));
        QUEUES_GENERATION.bump();
        q
    };
    match q {
        Some(q) => {
            q.close();
            true
        }
        None => false,
    }
}

/// Wait for events. `timeout_usec`: None = forever, Some(0) = poll.
/// Err: EBADF for a dead queue, ETIMEDOUT when nothing arrived.
pub(crate) fn wait(handle: u64, cap: usize, timeout_usec: Option<u32>) -> Result<Vec<KernelEvent>, HleResult> {
    let q = pin(handle).ok_or(ERR_BAD_QUEUE)?;
    let cap = cap.max(1);
    let out = match timeout_usec {
        Some(0) => {
            let mut s = lock_state(&q);
            q.collect(&mut s, cap)?
        }
        t => q.wait_for_events(cap, t)?,
    };
    if out.is_empty() { Err(ERR_TIMED_OUT) } else { Ok(out) }
}

/// Add a registration to a queue.
pub fn add_event(handle: u64, event: EqueueEvent) -> HleResult {
    pin(handle).map_or(ERR_BAD_QUEUE, |q| q.add_event(event))
}

/// Trigger `(ident, filter)` on one queue with `data`.
pub fn trigger(handle: u64, ident: u64, filter: i16, data: u64) -> HleResult {
    pin(handle).map_or(ERR_BAD_QUEUE, |q| q.trigger_event(ident, filter, data))
}

/// Delete `(ident, filter)` from one queue.
pub fn delete_event(handle: u64, ident: u64, filter: i16) -> HleResult {
    pin(handle).map_or(ERR_BAD_QUEUE, |q| q.delete_event(ident, filter))
}

/// True if `handle` names a live equeue.
pub fn is_valid_equeue(handle: u64) -> bool {
    pin(handle).is_some()
}

/// Register (or update) `(ident, filter)` with the standard hooks of its
/// filter.
pub fn register_event(handle: u64, ident: u64, filter: i16, udata: u64, flags: u16, low_water: u64) -> bool {
    let data = if filter == FILTER_GRAPHICS { ident } else { 0 };
    let event = EqueueEvent {
        event: KernelEvent { ident, filter, flags: flags | EV_ADD, fflags: 0, data, udata },
        filter: filter_for(filter, low_water),
        ..Default::default()
    };
    add_event(handle, event) == OK
}

/// Remove a registration.
pub fn delete_registered_event(handle: u64, ident: u64, filter: i16) -> bool {
    delete_event(handle, ident, filter) == OK
}

/// Trigger the `(ident, filter)` registration of one queue with `data`.
pub fn trigger_event(handle: u64, ident: u64, filter: i16, data: u64) -> bool {
    trigger(handle, ident, filter, data) == OK
}

/// AMPR completion for a registration added by `sceKernelAddAmprEvent`.
pub fn trigger_ampr_event(handle: u64, ident: u64, data: u64) -> bool {
    trigger_event(handle, ident, FILTER_AMPR, data)
}

/// Trigger `(ident, filter)` on every queue that registered it (GPU
/// end-of-pipe interrupts). Returns the number of queues triggered.
pub fn trigger_registered_events_by_filter(filter: i16, data: u64, ident: u64) -> usize {
    all_queues().iter().filter(|q| q.trigger_event(ident, filter, data) == OK).count()
}

/// Trigger a user event on every queue that has it.
pub fn trigger_user_event_for_all(id: u64, udata: u64) -> HleResult {
    if trigger_registered_events_by_filter(FILTER_USER, udata, id) > 0 { OK } else { ERR_NOT_FOUND }
}

fn process_millis() -> u64 {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_millis() as u64
}

/// Post a video-out event (flip, vblank, output mode) to `handle`.
/// `data` = (low 12 bits: time in ms) | (bits 12..15: occurrences since the
/// last delivery, saturating at 15) | (`hint` & !0xFFFF). The event is
/// edge-triggered; it is registered on first use.
pub fn trigger_display_event(handle: u64, ident: u64, filter: i16, hint: u64, udata: u64) -> bool {
    let Some(q) = pin(handle) else { return false };
    let mut s = lock_state(&q);
    if s.closed {
        return false;
    }
    let i = match Equeue::find(&mut s, ident, filter) {
        Some(i) => i,
        None => {
            s.events.push(EqueueEvent {
                event: KernelEvent { ident, filter, flags: EV_ADD | EV_CLEAR, ..Default::default() },
                ..Default::default()
            });
            s.events.len() - 1
        }
    };
    let e = &mut s.events[i];
    let count = if e.triggered { (((e.event.data >> 12) & 0xF) + 1).min(0xF) } else { 1 };
    e.event.data = (process_millis() & 0xFFF) | (count << 12) | (hint & 0xFFFF_FFFF_FFFF_0000);
    e.event.flags = EV_CLEAR;
    e.event.udata = udata;
    e.triggered = true;
    q.notify(&s);
    true
}

/// Number of deliverable occurrences on a queue (diagnostics / tests).
pub fn pending_count(handle: u64) -> usize {
    pin(handle).map_or(0, |q| {
        let s = lock_state(&q);
        s.events.iter().filter(|e| e.triggered).map(|e| 1 + e.pending.len()).sum()
    })
}

fn add_timer(handle: u64, id: i32, delay_ns: u64, periodic: bool, udata: u64) -> HleResult {
    let now = now_ns();
    let event = EqueueEvent {
        deadline_ns: now.saturating_add(delay_ns),
        interval_ns: if periodic { delay_ns } else { 0 },
        event: KernelEvent {
            ident: id as u32 as u64,
            filter: if periodic { FILTER_TIMER } else { FILTER_HR_TIMER },
            flags: EV_ADD | if periodic { EV_CLEAR } else { EV_ONESHOT },
            udata,
            ..Default::default()
        },
        ..Default::default()
    };
    add_event(handle, event)
}

// ---------------------------------------------------------------------------
// HLE callbacks
// ---------------------------------------------------------------------------

/// sceKernelCreateEqueue(SceKernelEqueue* eq, const char* name)
fn cb_create_equeue(ctx: &mut CpuContext) -> HleResult {
    let (out, name_ptr) = (ctx.rdi(), ctx.rsi());
    if out == 0 || name_ptr == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    let name = super::read_name(ctx, name_ptr, 32).unwrap_or_default();
    let handle = create_named(&name);
    if !ctx.write_u64(out, handle) {
        delete(handle);
        return ERR_MEMORY_FAULT;
    }
    log::debug!("[equeue] create '{name}' handle={handle:#x}");
    OK
}

/// sceKernelDeleteEqueue(eq)
fn cb_delete_equeue(ctx: &mut CpuContext) -> HleResult {
    if delete(ctx.rdi()) { OK } else { ERR_BAD_QUEUE }
}

/// sceKernelWaitEqueue(eq, SceKernelEvent* ev, int num, int* out, SceKernelUseconds* timeout)
fn cb_wait_equeue(ctx: &mut CpuContext) -> HleResult {
    let handle = ctx.rdi();
    let events = ctx.rsi();
    let cap = ctx.rdx() as i32;
    let out_count = ctx.rcx();
    let timeout_ptr = ctx.r8();
    if !is_valid_equeue(handle) {
        return ERR_BAD_QUEUE;
    }
    if events == 0 {
        return ERR_MEMORY_FAULT;
    }
    if cap < 1 {
        return ERR_INVALID_ARGUMENT;
    }
    let timeout = if timeout_ptr != 0 {
        match ctx.read_u32(timeout_ptr) {
            Some(t) => Some(t),
            None => return ERR_MEMORY_FAULT,
        }
    } else {
        None
    };
    match wait(handle, cap as usize, timeout) {
        Ok(evs) => {
            let mut delivered = 0u32;
            for (i, ev) in evs.iter().enumerate() {
                if !ctx.write_bytes(events + i as u64 * KERNEL_EVENT_SIZE, &ev.to_bytes()) {
                    break;
                }
                delivered += 1;
            }
            if out_count != 0 && !ctx.write_u32(out_count, delivered) {
                return ERR_MEMORY_FAULT;
            }
            if delivered == 0 { ERR_MEMORY_FAULT } else { OK }
        }
        Err(e) => {
            if out_count != 0 {
                ctx.write_u32(out_count, 0);
            }
            e
        }
    }
}

fn user_event(ident: u64, flags: u16) -> EqueueEvent {
    EqueueEvent {
        event: KernelEvent { ident, filter: FILTER_USER, flags, ..Default::default() },
        filter: filter_for(FILTER_USER, 0),
        ..Default::default()
    }
}

/// sceKernelAddUserEvent(eq, int id)
fn cb_add_user_event(ctx: &mut CpuContext) -> HleResult {
    add_event(ctx.rdi(), user_event(ctx.rsi() as i32 as u64, EV_ADD))
}

/// sceKernelAddUserEventEdge(eq, int id)
fn cb_add_user_event_edge(ctx: &mut CpuContext) -> HleResult {
    add_event(ctx.rdi(), user_event(ctx.rsi() as i32 as u64, EV_ADD | EV_CLEAR))
}

/// sceKernelDeleteUserEvent(eq, int id)
fn cb_delete_user_event(ctx: &mut CpuContext) -> HleResult {
    delete_event(ctx.rdi(), ctx.rsi() as i32 as u64, FILTER_USER)
}

/// sceKernelTriggerUserEvent(eq, int id, void* udata)
fn cb_trigger_user_event(ctx: &mut CpuContext) -> HleResult {
    trigger(ctx.rdi(), ctx.rsi() as i32 as u64, FILTER_USER, ctx.rdx())
}

/// sceKernelAddTimerEvent(eq, int id, SceKernelUseconds usec, void* udata)
fn cb_add_timer_event(ctx: &mut CpuContext) -> HleResult {
    add_timer(ctx.rdi(), ctx.rsi() as i32, ctx.rdx() as u32 as u64 * 1000, true, ctx.rcx())
}

/// sceKernelAddHRTimerEvent(eq, int id, SceKernelTimespec* ts, void* udata)
fn cb_add_hr_timer_event(ctx: &mut CpuContext) -> HleResult {
    let (handle, id, ts, udata) = (ctx.rdi(), ctx.rsi() as i32, ctx.rdx(), ctx.rcx());
    if ts == 0 {
        return ERR_MEMORY_FAULT;
    }
    let (Some(sec), Some(nsec)) = (ctx.read_u64(ts), ctx.read_u64(ts + 8)) else { return ERR_MEMORY_FAULT };
    let (sec, nsec) = (sec as i64, nsec as i64);
    if sec < 0 || !(0..1_000_000_000).contains(&nsec) || sec as u64 > (u64::MAX - nsec as u64) / 1_000_000_000 {
        return ERR_INVALID_ARGUMENT;
    }
    add_timer(handle, id, sec as u64 * 1_000_000_000 + nsec as u64, false, udata)
}

/// sceKernelDeleteHRTimerEvent(eq, int id)
fn cb_delete_hr_timer_event(ctx: &mut CpuContext) -> HleResult {
    delete_event(ctx.rdi(), ctx.rsi() as i32 as u32 as u64, FILTER_HR_TIMER)
}

/// sceKernelDeleteTimerEvent(eq, int id)
fn cb_delete_timer_event(ctx: &mut CpuContext) -> HleResult {
    delete_event(ctx.rdi(), ctx.rsi() as i32 as u32 as u64, FILTER_TIMER)
}

/// Register a level read or write event on a descriptor some readiness
/// source knows (ENOENT otherwise).
fn add_descriptor_event(ctx: &CpuContext, filter: i16) -> HleResult {
    let (handle, fd, low_water, udata) = (ctx.rdi(), ctx.rsi() as i32, ctx.rdx().max(1), ctx.rcx());
    if !is_valid_equeue(handle) {
        return ERR_BAD_QUEUE;
    }
    // The read end of a pipe takes a read event (its low-water mark does
    // not apply); a write event on a pipe stays kqueue-only: what the
    // console answers here is unknown.
    let pipe_read = filter == FILTER_READ && crate::filesystem::pipe::read_state(fd).is_some();
    if (crate::filesystem::pipe::is_pipe(fd) && !pipe_read) || descriptor_readiness(fd, filter, low_water).is_none() {
        return ERR_NOT_FOUND;
    }
    if register_event(handle, fd as u32 as u64, filter, udata, 0, low_water) { OK } else { ERR_BAD_QUEUE }
}

/// sceKernelAddReadEvent(eq, int fd, size_t low_water, void* udata)
fn cb_add_read_event(ctx: &mut CpuContext) -> HleResult {
    add_descriptor_event(ctx, FILTER_READ)
}

/// sceKernelDeleteReadEvent(eq, int fd)
fn cb_delete_read_event(ctx: &mut CpuContext) -> HleResult {
    delete_event(ctx.rdi(), ctx.rsi() as i32 as u32 as u64, FILTER_READ)
}

/// sceKernelAddWriteEvent(eq, int fd, size_t low_water, void* udata)
fn cb_add_write_event(ctx: &mut CpuContext) -> HleResult {
    add_descriptor_event(ctx, FILTER_WRITE)
}

/// sceKernelDeleteWriteEvent(eq, int fd)
fn cb_delete_write_event(ctx: &mut CpuContext) -> HleResult {
    delete_event(ctx.rdi(), ctx.rsi() as i32 as u32 as u64, FILTER_WRITE)
}

/// sceKernelAddAmprEvent(eq, int id, void* udata): always succeeds; an
/// invalid queue is ignored.
fn cb_add_ampr_event(ctx: &mut CpuContext) -> HleResult {
    let (handle, id, udata) = (ctx.rdi(), ctx.rsi() as i32 as u64, ctx.rdx());
    if handle != 0 {
        register_event(handle, id, FILTER_AMPR, udata, EV_ADD | EV_CLEAR, 0);
    }
    OK
}

/// sceKernelDeleteAmprEvent(eq, int id): always succeeds.
fn cb_delete_ampr_event(ctx: &mut CpuContext) -> HleResult {
    let (handle, id) = (ctx.rdi(), ctx.rsi() as i32 as u64);
    if handle != 0 {
        let _ = delete_event(handle, id, FILTER_AMPR);
    }
    OK
}

/// sceKernelGetKqueueFromEqueue(eq) -> kqueue (the handle itself)
fn cb_get_kqueue_from_equeue(ctx: &mut CpuContext) -> HleResult {
    ctx.rdi() as i64
}

/// sceKernelGetEventUserData(const SceKernelEvent*)
fn cb_get_event_user_data(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() == 0 {
        return 0;
    }
    ctx.read_u64(ctx.rdi().wrapping_add(0x18)).unwrap_or(0) as i64
}

/// sceKernelGetEventId(const SceKernelEvent*)
fn cb_get_event_id(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() == 0 {
        return 0;
    }
    ctx.read_u64(ctx.rdi()).unwrap_or(0) as i64
}

/// sceKernelGetEventFilter(const SceKernelEvent*) -> int (sign-extended i16)
fn cb_get_event_filter(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() == 0 {
        return 0;
    }
    let filter = ctx.read_bytes(ctx.rdi().wrapping_add(0x08), 2).map(|b| i16::from_le_bytes([b[0], b[1]])).unwrap_or(0);
    filter as i32 as u32 as i64
}

/// sceKernelGetEventData(const SceKernelEvent*)
fn cb_get_event_data(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() == 0 {
        return 0;
    }
    ctx.read_u64(ctx.rdi().wrapping_add(0x10)).unwrap_or(0) as i64
}

/// sceKernelGetEventFflags(const SceKernelEvent*)
fn cb_get_event_fflags(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() == 0 {
        return 0;
    }
    ctx.read_u32(ctx.rdi().wrapping_add(0x0C)).unwrap_or(0) as i64
}

/// sceKernelGetEventError(const SceKernelEvent*): data when EV_ERROR is set.
fn cb_get_event_error(ctx: &mut CpuContext) -> HleResult {
    let ev = ctx.rdi();
    if ev == 0 {
        return 0;
    }
    let flags = ctx.read_bytes(ev + 0x0A, 2).map(|b| u16::from_le_bytes([b[0], b[1]])).unwrap_or(0);
    if flags & EV_ERROR != 0 {
        ctx.read_u64(ev + 0x10).unwrap_or(0) as i32 as i64
    } else {
        0
    }
}

pub(crate) static EXPORTS: &[Export] = &[
    ("D0OdFMjp46I", "sceKernelCreateEqueue", "libKernel", cb_create_equeue),
    ("jpFjmgAC5AE", "sceKernelDeleteEqueue", "libKernel", cb_delete_equeue),
    ("fzyMKs9kim0", "sceKernelWaitEqueue", "libKernel", cb_wait_equeue),
    ("4R6-OvI2cEA", "sceKernelAddUserEvent", "libKernel", cb_add_user_event),
    ("WDszmSbWuDk", "sceKernelAddUserEventEdge", "libKernel", cb_add_user_event_edge),
    ("LJDwdSNTnDg", "sceKernelDeleteUserEvent", "libKernel", cb_delete_user_event),
    ("F6e0kwo4cnk", "sceKernelTriggerUserEvent", "libKernel", cb_trigger_user_event),
    ("57ZK+ODEXWY", "sceKernelAddTimerEvent", "libKernel", cb_add_timer_event),
    ("R74tt43xP6k", "sceKernelAddHRTimerEvent", "libKernel", cb_add_hr_timer_event),
    ("J+LF6LwObXU", "sceKernelDeleteHRTimerEvent", "libKernel", cb_delete_hr_timer_event),
    ("YWQFUyXIVdU", "sceKernelDeleteTimerEvent", "libKernel", cb_delete_timer_event),
    ("VHCS3rCd0PM", "sceKernelAddReadEvent", "libKernel", cb_add_read_event),
    ("JxJ4tfgKlXA", "sceKernelDeleteReadEvent", "libKernel", cb_delete_read_event),
    ("R-tyYMpYaxY", "sceKernelAddWriteEvent", "libKernel", cb_add_write_event),
    ("cBGTk8S92XM", "sceKernelDeleteWriteEvent", "libKernel", cb_delete_write_event),
    ("bBfz7kMF2Ho", "sceKernelAddAmprEvent", "libKernel", cb_add_ampr_event),
    ("vuae5JPNt9A", "sceKernelAddAmprSystemEvent", "libKernel", cb_add_ampr_event),
    ("bMmid3pfyjo", "sceKernelDeleteAmprEvent", "libKernel", cb_delete_ampr_event),
    ("Ij+ryuEClXQ", "sceKernelDeleteAmprSystemEvent", "libKernel", cb_delete_ampr_event),
    ("QyrxcdBrb0M", "sceKernelGetKqueueFromEqueue", "libKernel", cb_get_kqueue_from_equeue),
    ("vz+pg2zdopI", "sceKernelGetEventUserData", "libKernel", cb_get_event_user_data),
    ("mJ7aghmgvfc", "sceKernelGetEventId", "libKernel", cb_get_event_id),
    ("23CPPI1tyBY", "sceKernelGetEventFilter", "libKernel", cb_get_event_filter),
    ("kwGyyjohI50", "sceKernelGetEventData", "libKernel", cb_get_event_data),
    ("Q0qr9AyqJSk", "sceKernelGetEventFflags", "libKernel", cb_get_event_fflags),
    ("Uu-iDFC9aUc", "sceKernelGetEventError", "libKernel", cb_get_event_error),
];

#[cfg(test)]
mod tests;
