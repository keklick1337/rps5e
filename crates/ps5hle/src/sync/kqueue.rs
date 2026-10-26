// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! kqueue(2) / kevent(2) on top of the kernel event queues, vnode watches
//! (EVFILT_VNODE: inotify on the host) and sceKernelAddFileEvent /
//! sceKernelDeleteFileEvent.
//!
//! A kqueue is an event queue behind a descriptor number of the shared
//! descriptor space. kevent applies the change list in order (a failed
//! change is reported as an EV_ERROR event when there is room, else the
//! call fails with its errno) and then collects triggered events.
//!
//! Filters: EVFILT_READ (kernel sockets, the read end of a pipe),
//! EVFILT_WRITE (kernel sockets), EVFILT_VNODE (host files and
//! directories), EVFILT_TIMER and EVFILT_USER. A request whose console
//! behaviour is unknown (another filter, EV_CLEAR on a descriptor,
//! changing a timer, a socket low-water mark the reference refuses) is
//! logged and refused with EINVAL.

#[cfg(windows)]
#[allow(unused_imports)]
use crate::host::c as libc;
use super::equeue::{self, Equeue, EqueueEvent, Filter, KernelEvent};
use super::equeue::{EV_ADD, EV_CLEAR, EV_ERROR, EV_ONESHOT};
use super::equeue::{FILTER_FILE, FILTER_READ, FILTER_TIMER, FILTER_USER, FILTER_WRITE};
use super::{Export, ERR_NOT_FOUND, OK};
use crate::context::CpuContext;
use crate::registry::HleResult;
use std::collections::HashMap;
#[cfg(unix)]
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
#[cfg_attr(windows, allow(unused_imports))]
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
#[cfg_attr(windows, allow(unused_imports))]
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const ENOENT: i32 = 2;
const EBADF: i32 = 9;
const EFAULT: i32 = 14;
const EINVAL: i32 = 22;
const EMFILE: i32 = 24;
const EPIPE: i32 = 32;

const EV_DELETE: u16 = 0x0002;
const EV_ENABLE: u16 = 0x0004;
const EV_DISABLE: u16 = 0x0008;
const EV_RECEIPT: u16 = 0x0040;
const EV_FORCEONESHOT: u16 = 0x0100;
const EV_SYSFLAGS: u16 = 0xF000;
/// Action flags never kept on a registration.
const KNOTE_CLEARED_FLAGS: u16 = EV_ADD | EV_DELETE | EV_ENABLE | EV_DISABLE | EV_FORCEONESHOT;

/// Events copied out per call when the caller passes no event list.
const KQ_NEVENTS: usize = 8;

const NOTE_FFCTRLMASK: u32 = 0xC000_0000;
const NOTE_FFLAGSMASK: u32 = 0x00FF_FFFF;
const NOTE_FFAND: u32 = 0x4000_0000;
const NOTE_FFOR: u32 = 0x8000_0000;
const NOTE_FFCOPY: u32 = 0xC000_0000;
const NOTE_TRIGGER: u32 = 0x0100_0000;
const NOTE_LOWAT: u32 = 0x0001;
const NOTE_SECONDS: u32 = 0x0001;
const NOTE_MSECONDS: u32 = 0x0002;
const NOTE_USECONDS: u32 = 0x0004;
const NOTE_NSECONDS: u32 = 0x0008;
const NOTE_TIMER_PRECMASK: u32 = NOTE_SECONDS | NOTE_MSECONDS | NOTE_USECONDS | NOTE_NSECONDS;

pub(crate) const NOTE_DELETE: u32 = 0x0001;
pub(crate) const NOTE_WRITE: u32 = 0x0002;
pub(crate) const NOTE_EXTEND: u32 = 0x0004;
pub(crate) const NOTE_ATTRIB: u32 = 0x0008;
pub(crate) const NOTE_LINK: u32 = 0x0010;
pub(crate) const NOTE_RENAME: u32 = 0x0020;
const MAPPED_NOTES: u32 = NOTE_DELETE | NOTE_WRITE | NOTE_EXTEND | NOTE_ATTRIB | NOTE_LINK | NOTE_RENAME;

/// kqueue descriptor -> event queue handle.
static KQUEUES: Mutex<Option<HashMap<i32, u64>>> = Mutex::new(None);

fn kq_handle(fd: i32) -> Option<u64> {
    KQUEUES.lock().unwrap_or_else(|p| p.into_inner()).as_ref()?.get(&fd).copied()
}

/// True when `fd` is a kqueue.
pub(crate) fn is_kqueue(fd: i32) -> bool {
    kq_handle(fd).is_some()
}

/// True when kqueue `fd` holds triggered events (select and poll report
/// it readable then).
pub(crate) fn has_pending(fd: i32) -> bool {
    kq_handle(fd).is_some_and(|h| equeue::pending_count(h) > 0)
}

/// close() hook: true when `fd` was a kqueue (now closed).
pub(crate) fn close(fd: i32) -> bool {
    let handle = KQUEUES.lock().unwrap_or_else(|p| p.into_inner()).as_mut().and_then(|m| m.remove(&fd));
    match handle {
        Some(h) => {
            equeue::delete(h);
            crate::filesystem::release_fd(fd);
            true
        }
        None => false,
    }
}

/// A request the console's answer is unknown for.
fn unsupported(what: &str) -> i32 {
    log::error!("[kqueue] {what}: not supported");
    EINVAL
}

/// Kernel error code -> errno.
fn errno_of(r: HleResult) -> i32 {
    if r == OK { 0 } else { (r as u32 & 0xFFFF) as i32 }
}

fn fail(ctx: &CpuContext, e: i32) -> HleResult {
    crate::filesystem::set_errno(ctx, e);
    -1
}

/// kqueue(): a new queue on a fresh descriptor.
fn cb_kqueue(ctx: &mut CpuContext) -> HleResult {
    let fd = crate::filesystem::reserve_fd();
    if fd < 0 {
        return fail(ctx, EMFILE);
    }
    let handle = equeue::create_named("kqueue");
    KQUEUES.lock().unwrap_or_else(|p| p.into_inner()).get_or_insert_with(HashMap::new).insert(fd, handle);
    log::debug!("[kqueue] kqueue -> {fd}");
    fd as HleResult
}

fn event_from_bytes(b: &[u8]) -> KernelEvent {
    let u64_at = |o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
    KernelEvent {
        ident: u64_at(0),
        filter: i16::from_le_bytes([b[8], b[9]]),
        flags: u16::from_le_bytes([b[10], b[11]]),
        fflags: u32::from_le_bytes(b[12..16].try_into().unwrap()),
        data: u64_at(16),
        udata: u64_at(24),
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Errno of a read/write registration on `ident` in kqueue `kq`.
fn descriptor_filter_error(kq: i32, ident: u64) -> i32 {
    if ident > i32::MAX as u64 {
        return EBADF;
    }
    let fd = ident as i32;
    if crate::net::is_kernel_socket(fd) {
        return 0;
    }
    if !crate::filesystem::descriptor_open(fd) {
        return EBADF;
    }
    if fd == kq { EINVAL } else { 0 }
}

/// NOTE_LOWAT of a read or write change (sockets only).
fn descriptor_low_water(change: &KernelEvent) -> Result<u64, i32> {
    let fd = change.ident as i32;
    if change.fflags & NOTE_LOWAT == 0 || !crate::net::is_kernel_socket(fd) {
        return Ok(0);
    }
    let data = change.data as i64;
    if data < 1 {
        return Err(unsupported("kevent: NOTE_LOWAT below 1 on a socket"));
    }
    if data > 1 && crate::net::kernel_socket_type(fd) != Some(1) {
        return Err(unsupported("kevent: NOTE_LOWAT above 1 on a socket that is not a stream"));
    }
    Ok(data as u64)
}

/// Level descriptor events stay triggered while the descriptor is ready.
fn descriptor_reset(e: &mut EqueueEvent) {
    e.triggered = false;
}

fn add_descriptor_knote(q: &Equeue, change: &mut KernelEvent, flags: u16) -> i32 {
    let fd = change.ident as i32;
    if is_kqueue(fd) {
        if change.filter != FILTER_READ {
            return EINVAL;
        }
        return unsupported("kevent: EVFILT_READ on a kqueue descriptor");
    }
    let socket = crate::net::is_kernel_socket(fd);
    let pipe_state = if socket {
        None
    } else if change.filter == FILTER_READ {
        crate::filesystem::pipe::read_state(fd).map(|_| false)
    } else {
        crate::filesystem::pipe::write_state(fd).map(|(_, eof, _)| eof)
    };
    let pipe = pipe_state.is_some();
    // Standard input of a console process: a read event is accepted and
    // never fires (there is no terminal input).
    if fd == 0 && !socket && !pipe && change.filter == FILTER_READ {
        let event = EqueueEvent {
            event: KernelEvent { ident: 0, filter: change.filter, flags, fflags: 0, data: 0, udata: change.udata },
            ..Default::default()
        };
        change.fflags = 0;
        change.data = 0;
        return errno_of(q.add_event(event));
    }
    if !socket && !pipe {
        return unsupported(if change.filter == FILTER_READ {
            "kevent: EVFILT_READ on a descriptor that is neither a socket nor the read end of a pipe"
        } else {
            "kevent: EVFILT_WRITE on a descriptor that is neither a socket nor the write end of a pipe"
        });
    }
    // A write registration on a pipe whose reader is gone.
    if pipe_state == Some(true) {
        return EPIPE;
    }
    let low_water = match descriptor_low_water(change) {
        Ok(v) => v,
        Err(e) => return e,
    };
    if flags & EV_CLEAR != 0 {
        return unsupported(if pipe { "kevent: EV_CLEAR on a pipe" } else { "kevent: EV_CLEAR on a socket" });
    }
    let event = EqueueEvent {
        event: KernelEvent { ident: change.ident, filter: change.filter, flags, fflags: 0, data: 0, udata: change.udata },
        filter: Filter { low_water, reset: Some(descriptor_reset), ..Default::default() },
        ..Default::default()
    };
    change.fflags = 0;
    change.data = 0;
    errno_of(q.add_event(event))
}

/// Period of an EVFILT_TIMER in nanoseconds (milliseconds by default).
fn timer_period_ns(data: u64, fflags: u32) -> Option<u64> {
    let unit = match fflags & NOTE_TIMER_PRECMASK {
        NOTE_SECONDS => 1_000_000_000,
        NOTE_MSECONDS | 0 => 1_000_000,
        NOTE_USECONDS => 1_000,
        NOTE_NSECONDS => 1,
        _ => return None,
    };
    Some(data.saturating_mul(unit))
}

fn add_timer_knote(q: &Equeue, change: &mut KernelEvent, flags: u16) -> i32 {
    let mut period = change.data as i64;
    if period < 0 {
        return EINVAL;
    }
    if period == 0 && flags & EV_ONESHOT == 0 {
        period = 1;
    }
    if change.fflags & !NOTE_TIMER_PRECMASK != 0 {
        return EINVAL;
    }
    let Some(period_ns) = timer_period_ns(period as u64, change.fflags) else { return EINVAL };
    let event = EqueueEvent {
        deadline_ns: equeue::now_ns().saturating_add(period_ns),
        interval_ns: if flags & EV_ONESHOT != 0 { 0 } else { period_ns },
        event: KernelEvent { flags: flags | EV_CLEAR, fflags: 0, data: 0, ..*change },
        ..Default::default()
    };
    change.fflags = 0;
    change.data = 0;
    errno_of(q.add_event(event))
}

/// User events keep their fflags and data across deliveries; EV_CLEAR
/// only drops the trigger.
fn user_reset(e: &mut EqueueEvent) {
    if e.event.flags & EV_CLEAR != 0 {
        e.triggered = false;
    }
}

fn add_user_knote(q: &Equeue, change: &mut KernelEvent, flags: u16) -> i32 {
    let event = EqueueEvent {
        event: KernelEvent { flags, ..*change },
        filter: Filter { reset: Some(user_reset), ..Default::default() },
        ..Default::default()
    };
    change.fflags = 0;
    change.data = 0;
    errno_of(q.add_event(event))
}

/// NOTE_TRIGGER and the fflags control of an EVFILT_USER change.
fn touch_user_knote(e: &mut EqueueEvent, change: &KernelEvent) {
    if change.fflags & NOTE_TRIGGER != 0 {
        e.triggered = true;
    }
    let fflags = change.fflags & NOTE_FFLAGSMASK;
    match change.fflags & NOTE_FFCTRLMASK {
        NOTE_FFAND => e.event.fflags &= fflags,
        NOTE_FFOR => e.event.fflags |= fflags,
        NOTE_FFCOPY => e.event.fflags = fflags,
        _ => {}
    }
    e.event.data = change.data;
    if change.flags & EV_CLEAR != 0 {
        e.triggered = false;
    }
}

fn modify_knote(q: &Equeue, change: &KernelEvent) -> i32 {
    if change.filter == FILTER_TIMER {
        // A change that carries no new period only enables, disables or
        // retags the note: the interval stays and a disabled timer keeps
        // its phase, so the first delivery after EV_ENABLE sums the
        // expiries it missed.
        if change.data == 0 && change.fflags == 0 {
            let c = *change;
            return errno_of(q.edit_event(c.ident, c.filter, |e| {
                e.event.udata = c.udata;
                if c.flags & EV_ENABLE != 0 {
                    e.disabled = false;
                }
                if c.flags & EV_DISABLE != 0 {
                    e.disabled = true;
                }
            }));
        }
        // A period change without EV_ADD is refused. Re-adding an
        // existing timer first fires the periods that have expired and
        // then takes the new interval (measured on hardware).
        if change.flags & EV_ADD == 0 {
            return EINVAL;
        }
        let mut readd = *change;
        let flags = readd.flags & !KNOTE_CLEARED_FLAGS;
        readd.flags |= EV_ADD;
        return add_timer_knote(q, &mut readd, flags);
    }
    let descriptor = change.filter == FILTER_READ || change.filter == FILTER_WRITE;
    let low_water = if descriptor {
        match descriptor_low_water(change) {
            Ok(v) => v,
            Err(e) => return e,
        }
    } else {
        0
    };
    if change.filter == FILTER_FILE {
        if let Err(e) = require_vnode_notes(change.fflags) {
            return e;
        }
    }
    let c = *change;
    errno_of(q.edit_event(c.ident, c.filter, |e| {
        e.event.udata = c.udata;
        if descriptor {
            e.filter.low_water = low_water;
        }
        if c.flags & EV_ENABLE != 0 {
            e.disabled = false;
        }
        if c.flags & EV_DISABLE != 0 {
            e.disabled = true;
        }
        if c.flags & EV_FORCEONESHOT != 0 {
            e.event.flags |= EV_ONESHOT;
            e.triggered = true;
        }
        if c.filter == FILTER_USER {
            touch_user_knote(e, &c);
        } else if c.filter == FILTER_FILE {
            e.filter.notes = c.fflags;
        }
    }))
}

/// Apply one change; `change` is updated as the EV_ERROR / receipt event
/// reports it. Returns the errno (0 = done).
fn register(q: &Equeue, kq: i32, change: &mut KernelEvent) -> i32 {
    if change.flags & (EV_ENABLE | EV_DISABLE) == EV_ENABLE | EV_DISABLE {
        return EINVAL;
    }
    if change.filter > 0 {
        return EINVAL;
    }
    let known = [FILTER_READ, FILTER_WRITE, FILTER_FILE, FILTER_TIMER, FILTER_USER];
    if !known.contains(&change.filter) {
        return unsupported(&format!("kevent: filter {}", change.filter));
    }
    if matches!(change.filter, FILTER_READ | FILTER_WRITE | FILTER_FILE) {
        let e = descriptor_filter_error(kq, change.ident);
        if e != 0 {
            return e;
        }
    }
    if !q.has_event(change.ident, change.filter) {
        if change.flags & EV_ADD == 0 {
            return ENOENT;
        }
        let flags = change.flags & !KNOTE_CLEARED_FLAGS;
        let e = match change.filter {
            FILTER_TIMER => add_timer_knote(q, change, flags),
            FILTER_USER => add_user_knote(q, change, flags),
            FILTER_FILE => {
                let notes = change.fflags;
                change.fflags = 0;
                change.data = 0;
                add_vnode_event(q, change.ident as i32, notes, flags, change.udata)
            }
            _ => add_descriptor_knote(q, change, flags),
        };
        if e == 0 && change.flags & EV_DISABLE != 0 {
            q.edit_event(change.ident, change.filter, |e| e.disabled = true);
        }
        return e;
    }
    if change.flags & EV_DELETE != 0 {
        return errno_of(q.delete_event(change.ident, change.filter));
    }
    modify_knote(q, change)
}

/// kevent(kq, changelist, nchanges, eventlist, nevents, timeout)
fn cb_kevent(ctx: &mut CpuContext) -> HleResult {
    let (kq, changes, nchanges) = (ctx.rdi() as i32, ctx.rsi(), ctx.rdx() as i32);
    let (mut out, mut nevents, timeout) = (ctx.rcx(), ctx.r8() as i32, ctx.r9());
    let Some(q) = kq_handle(kq).and_then(equeue::pin) else { return fail(ctx, EBADF) };
    if nevents < 0 {
        return fail(ctx, EINVAL);
    }
    if nchanges > 0 && changes == 0 {
        return fail(ctx, EFAULT);
    }
    let mut errors = 0i64;
    for i in 0..nchanges.max(0) as u64 {
        let Some(bytes) = ctx.read_bytes(changes + i * equeue::KERNEL_EVENT_SIZE, equeue::KERNEL_EVENT_SIZE as usize) else {
            return fail(ctx, EFAULT);
        };
        let mut change = event_from_bytes(&bytes);
        if change.filter == 0 {
            continue;
        }
        change.flags &= !EV_SYSFLAGS;
        let e = register(&q, kq, &mut change);
        if e == 0 && change.flags & EV_RECEIPT == 0 {
            continue;
        }
        if nevents == 0 {
            return if e == 0 { 0 } else { fail(ctx, e) };
        }
        change.flags = EV_ERROR;
        change.data = e as u64;
        if out != 0 {
            if !ctx.write_bytes(out, &change.to_bytes()) {
                return fail(ctx, EFAULT);
            }
            out += equeue::KERNEL_EVENT_SIZE;
        }
        nevents -= 1;
        errors += 1;
    }
    if errors != 0 {
        return errors;
    }
    if nevents == 0 {
        return 0;
    }
    let mut deadline = None;
    if timeout != 0 {
        let (Some(sec), Some(nsec)) = (ctx.read_u64(timeout), ctx.read_u64(timeout + 8)) else { return fail(ctx, EFAULT) };
        let (sec, nsec) = (sec as i64, nsec as i64);
        if sec < 0 || !(0..1_000_000_000).contains(&nsec) {
            return fail(ctx, EINVAL);
        }
        // A wait beyond 2^31 seconds is a wait without end.
        if sec <= i32::MAX as i64 {
            deadline = Some(Instant::now() + Duration::new(sec as u64, nsec as u32));
        }
    }
    let cap = if out != 0 { nevents as usize } else { (nevents as usize).min(KQ_NEVENTS) };
    let events = match q.wait_until(cap, deadline) {
        Ok(v) => v,
        Err(e) => return fail(ctx, errno_of(e)),
    };
    if events.is_empty() {
        return 0;
    }
    if out == 0 {
        return fail(ctx, EFAULT);
    }
    for (i, ev) in events.iter().enumerate() {
        if !ctx.write_bytes(out + i as u64 * equeue::KERNEL_EVENT_SIZE, &ev.to_bytes()) {
            return fail(ctx, EFAULT);
        }
    }
    events.len() as HleResult
}

// ---------------------------------------------------------------------------
// Vnode watches
// ---------------------------------------------------------------------------

fn require_vnode_notes(notes: u32) -> Result<(), i32> {
    if notes & !MAPPED_NOTES != 0 {
        return Err(unsupported(&format!("EVFILT_VNODE: fflags {:#x}", notes & !MAPPED_NOTES)));
    }
    Ok(())
}

/// Errno of a vnode watch of `fd`.
fn vnode_descriptor_error(fd: i32) -> i32 {
    if fd < 0 {
        return EBADF;
    }
    if crate::net::is_kernel_socket(fd) {
        return EINVAL;
    }
    if !crate::filesystem::descriptor_open(fd) {
        return EBADF;
    }
    if is_kqueue(fd) { EINVAL } else { 0 }
}

static NEXT_WATCH: AtomicU64 = AtomicU64::new(1);

/// Register an EVFILT_VNODE watch of `fd` on `q`. Windows has no host
/// watch yet: the registration is kept and never fires.
#[cfg(windows)]
fn add_vnode_event(q: &Equeue, fd: i32, notes: u32, flags: u16, udata: u64) -> i32 {
    if let Err(e) = require_vnode_notes(notes) {
        return e;
    }
    let e = vnode_descriptor_error(fd);
    if e != 0 {
        return e;
    }
    match crate::filesystem::watch_target(fd) {
        None => return EBADF,
        Some(crate::filesystem::WatchTarget::Fifo) => return EINVAL,
        Some(crate::filesystem::WatchTarget::Device) => {
            return unsupported("EVFILT_VNODE: a descriptor that is not a file or a directory");
        }
        Some(_) => {}
    }
    let id = NEXT_WATCH.fetch_add(1, Ordering::Relaxed);
    let event = EqueueEvent {
        event: KernelEvent { ident: fd as u32 as u64, filter: FILTER_FILE, flags, fflags: 0, data: 0, udata },
        filter: Filter { watch: id, notes, ..Default::default() },
        ..Default::default()
    };
    let r = q.add_event(event);
    if r != OK {
        return errno_of(r);
    }
    0
}


/// Stop switch of one watch thread.
#[cfg(unix)]
struct WatchStop {
    stopped: AtomicBool,
    event: OwnedFd,
}

#[cfg(unix)]
impl WatchStop {
    fn stop(&self) {
        self.stopped.store(true, Ordering::Release);
        let one = 1u64;
        // SAFETY: an eventfd takes one 8-byte counter increment.
        unsafe { libc::write(self.event.as_raw_fd(), &one as *const u64 as *const libc::c_void, 8) };
    }
}

/// Register an EVFILT_VNODE watch of `fd` on `q` and start its host watch.
#[cfg(unix)]
fn add_vnode_event(q: &Equeue, fd: i32, notes: u32, flags: u16, udata: u64) -> i32 {
    if let Err(e) = require_vnode_notes(notes) {
        return e;
    }
    let e = vnode_descriptor_error(fd);
    if e != 0 {
        return e;
    }
    let host = match crate::filesystem::watch_target(fd) {
        None => return EBADF,
        Some(crate::filesystem::WatchTarget::Fifo) => return EINVAL,
        Some(crate::filesystem::WatchTarget::Device) => {
            return unsupported("EVFILT_VNODE: a descriptor that is not a file or a directory");
        }
        // The /app0 image never changes: the watch never fires.
        Some(crate::filesystem::WatchTarget::Image) => None,
        Some(crate::filesystem::WatchTarget::File(raw)) => match HostWatch::open(&format!("/proc/self/fd/{raw}")) {
            Ok(h) => Some(h),
            Err(e) => return e,
        },
        Some(crate::filesystem::WatchTarget::Dir(path)) => match HostWatch::open(&path.to_string_lossy()) {
            Ok(h) => Some(h),
            Err(e) => return e,
        },
    };
    // SAFETY: a fresh eventfd owned by the watch.
    let raw = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC | libc::EFD_NONBLOCK) };
    if raw < 0 {
        return unsupported("EVFILT_VNODE: eventfd failed");
    }
    let stop = Arc::new(WatchStop { stopped: AtomicBool::new(false), event: unsafe { OwnedFd::from_raw_fd(raw) } });
    let id = NEXT_WATCH.fetch_add(1, Ordering::Relaxed);
    let hook_stop = stop.clone();
    let event = EqueueEvent {
        event: KernelEvent { ident: fd as u32 as u64, filter: FILTER_FILE, flags, fflags: 0, data: 0, udata },
        filter: Filter {
            watch: id,
            notes,
            on_delete: Some(Arc::new(move |_, _| hook_stop.stop())),
            ..Default::default()
        },
        ..Default::default()
    };
    let r = q.add_event(event);
    if r != OK {
        return errno_of(r);
    }
    if let Some(host) = host {
        let (handle, ident) = (q.handle(), fd as u32 as u64);
        let spawned = std::thread::Builder::new()
            .name(format!("vnode-watch-{fd}"))
            .spawn(move || host.run(&stop, |hint| deliver_vnode(handle, ident, id, hint)));
        if spawned.is_err() {
            log::error!("[kqueue] EVFILT_VNODE: cannot start the watch thread of {fd}");
        }
    }
    0
}

/// A host change with `hint` notes reached watch `id` of `(handle, ident)`.
#[cfg(unix)]
fn deliver_vnode(handle: u64, ident: u64, id: u64, hint: u32) {
    if hint == 0 {
        return;
    }
    let Some(q) = equeue::pin(handle) else { return };
    let _ = q.edit_event(ident, FILTER_FILE, |e| {
        if e.filter.watch != id || e.disabled {
            return;
        }
        if e.filter.notes & hint != 0 {
            e.event.fflags |= hint;
        }
        e.triggered = e.event.fflags != 0;
    });
}

/// inotify watch of one file or directory.
#[cfg(unix)]
struct HostWatch {
    notify: OwnedFd,
    /// O_PATH descriptor of the watched object (follows renames).
    target: OwnedFd,
    /// Directory stream of a watched directory (settles renames).
    entries: Option<OwnedFd>,
    directory: bool,
    last: libc::stat,
}

#[cfg(unix)]
fn fstat(fd: &OwnedFd) -> Option<libc::stat> {
    // SAFETY: fstat fills a zeroed stat.
    unsafe {
        let mut st: libc::stat = std::mem::zeroed();
        (libc::fstat(fd.as_raw_fd(), &mut st) == 0).then_some(st)
    }
}

#[cfg(unix)]
fn open_path(path: &str, flags: i32) -> Option<OwnedFd> {
    let c = std::ffi::CString::new(path).ok()?;
    // SAFETY: a NUL-terminated path; the descriptor is owned on success.
    let fd = unsafe { libc::open(c.as_ptr(), flags | libc::O_CLOEXEC) };
    (fd >= 0).then(|| unsafe { OwnedFd::from_raw_fd(fd) })
}

#[cfg(unix)]
impl HostWatch {
    fn open(path: &str) -> Result<Self, i32> {
        let target = open_path(path, libc::O_PATH).ok_or(EBADF)?;
        let last = fstat(&target).ok_or(EBADF)?;
        let kind = last.st_mode & libc::S_IFMT;
        if kind == libc::S_IFIFO || kind == libc::S_IFSOCK {
            return Err(EINVAL);
        }
        if kind != libc::S_IFREG && kind != libc::S_IFDIR {
            return Err(unsupported("EVFILT_VNODE: a descriptor that is not a file or a directory"));
        }
        let directory = kind == libc::S_IFDIR;
        let entries = if directory {
            Some(open_path(path, libc::O_RDONLY | libc::O_DIRECTORY).ok_or_else(|| unsupported("EVFILT_VNODE: opening the watched directory"))?)
        } else {
            None
        };
        // SAFETY: a fresh inotify instance owned by the watch.
        let raw = unsafe { libc::inotify_init1(libc::IN_CLOEXEC | libc::IN_NONBLOCK) };
        if raw < 0 {
            return Err(unsupported("EVFILT_VNODE: inotify_init1 failed"));
        }
        let notify = unsafe { OwnedFd::from_raw_fd(raw) };
        let mut mask = libc::IN_MODIFY | libc::IN_ATTRIB | libc::IN_MOVE_SELF | libc::IN_DELETE_SELF;
        if directory {
            mask |= libc::IN_CREATE | libc::IN_DELETE | libc::IN_MOVED_FROM | libc::IN_MOVED_TO;
        }
        let c = std::ffi::CString::new(path).map_err(|_| EINVAL)?;
        // SAFETY: a NUL-terminated path on our inotify instance.
        if unsafe { libc::inotify_add_watch(notify.as_raw_fd(), c.as_ptr(), mask) } < 0 {
            return Err(unsupported("EVFILT_VNODE: inotify_add_watch failed"));
        }
        Ok(Self { notify, target, entries, directory, last })
    }

    fn attribute_hint(&mut self) -> u32 {
        let Some(cur) = fstat(&self.target) else { return NOTE_ATTRIB };
        let mut hint = NOTE_ATTRIB;
        if cur.st_nlink > self.last.st_nlink {
            hint = NOTE_LINK;
        } else if cur.st_nlink < self.last.st_nlink {
            hint = if self.directory && cur.st_nlink != 0 { NOTE_LINK } else { NOTE_DELETE };
        }
        self.last = cur;
        hint
    }

    fn modify_hint(&mut self) -> u32 {
        let Some(cur) = fstat(&self.target) else { return NOTE_WRITE };
        let mut hint = NOTE_WRITE;
        if cur.st_size > self.last.st_size {
            hint = NOTE_WRITE | NOTE_EXTEND;
        } else if cur.st_size < self.last.st_size {
            hint = NOTE_ATTRIB;
        }
        self.last = cur;
        hint
    }

    /// Drain the inotify queue: (mask, cookie) of each event.
    fn read_events(&self, out: &mut Vec<(u32, u32)>) {
        let mut buf = [0u8; 16384];
        loop {
            // SAFETY: reads into a local buffer.
            let n = unsafe { libc::read(self.notify.as_raw_fd(), buf.as_mut_ptr() as *mut libc::c_void, buf.len()) };
            if n <= 0 {
                return;
            }
            let mut off = 0usize;
            while off + 16 <= n as usize {
                let mask = u32::from_le_bytes(buf[off + 4..off + 8].try_into().unwrap());
                let cookie = u32::from_le_bytes(buf[off + 8..off + 12].try_into().unwrap());
                let len = u32::from_le_bytes(buf[off + 12..off + 16].try_into().unwrap()) as usize;
                out.push((mask, cookie));
                off += 16 + len;
            }
        }
    }

    /// Read events until no rename is half reported (a move out of the
    /// directory has one half; listing the directory lets a move inside it
    /// finish).
    fn read_settled(&self, out: &mut Vec<(u32, u32)>) {
        self.read_events(out);
        let mut settled = 0;
        for _ in 0..64 {
            if !has_unpaired_move(out, settled) {
                return;
            }
            settled = out.len();
            if let Some(entries) = &self.entries {
                let mut buf = [0u8; 1024];
                // SAFETY: getdents64 into a local buffer.
                unsafe { libc::syscall(libc::SYS_getdents64, entries.as_raw_fd(), buf.as_mut_ptr(), buf.len()) };
            }
            self.read_events(out);
            if out.len() == settled {
                return;
            }
        }
    }

    fn run(mut self, stop: &WatchStop, deliver: impl Fn(u32)) {
        let mut fds = [
            libc::pollfd { fd: self.notify.as_raw_fd(), events: libc::POLLIN, revents: 0 },
            libc::pollfd { fd: stop.event.as_raw_fd(), events: libc::POLLIN, revents: 0 },
        ];
        loop {
            if stop.stopped.load(Ordering::Acquire) || ps5vm::shutdown::requested() {
                return;
            }
            // SAFETY: two valid pollfds; bounded so a shutdown is noticed.
            let r = unsafe { libc::poll(fds.as_mut_ptr(), 2, 100) };
            if r <= 0 || fds[1].revents != 0 {
                continue;
            }
            let mut events = Vec::new();
            self.read_settled(&mut events);
            let moves = count_moves(&events);
            let mut hints = Vec::new();
            let mut watching = true;
            for &(mask, cookie) in &events {
                let link = if mask & libc::IN_ISDIR != 0 { NOTE_LINK } else { 0 };
                if mask & libc::IN_MODIFY != 0 {
                    hints.push(self.modify_hint());
                }
                if mask & libc::IN_ATTRIB != 0 {
                    hints.push(self.attribute_hint());
                }
                if mask & (libc::IN_CREATE | libc::IN_DELETE) != 0 {
                    hints.push(NOTE_WRITE | link);
                }
                if mask & (libc::IN_MOVED_FROM | libc::IN_MOVED_TO) != 0 {
                    let paired = moves.get(&cookie).copied().unwrap_or(0) > 1;
                    hints.push(if paired { NOTE_WRITE } else { NOTE_WRITE | NOTE_EXTEND | link });
                }
                if mask & libc::IN_MOVE_SELF != 0 {
                    hints.push(NOTE_RENAME);
                }
                if mask & libc::IN_DELETE_SELF != 0 {
                    hints.push(NOTE_DELETE);
                }
                if mask & libc::IN_IGNORED != 0 {
                    watching = false;
                }
            }
            for h in hints {
                deliver(h);
            }
            if !watching {
                return;
            }
        }
    }
}

#[cfg(unix)]
fn count_moves(events: &[(u32, u32)]) -> HashMap<u32, u32> {
    let mut moves = HashMap::new();
    for &(mask, cookie) in events {
        if mask & (libc::IN_MOVED_FROM | libc::IN_MOVED_TO) != 0 {
            *moves.entry(cookie).or_insert(0) += 1;
        }
    }
    moves
}

#[cfg(unix)]
fn has_unpaired_move(events: &[(u32, u32)], first: usize) -> bool {
    let moves = count_moves(events);
    events[first..].iter().any(|&(mask, cookie)| mask & libc::IN_MOVED_FROM != 0 && moves.get(&cookie) == Some(&1))
}

// ---------------------------------------------------------------------------
// sceKernelAddFileEvent / sceKernelDeleteFileEvent
// ---------------------------------------------------------------------------

/// sceKernelAddFileEvent(eq, int fd, int watch, void* udata): an edge
/// vnode registration; an existing one takes the new notes and user data.
fn cb_add_file_event(ctx: &mut CpuContext) -> HleResult {
    let (handle, fd, notes, udata) = (ctx.rdi(), ctx.rsi() as i32, ctx.rdx() as u32, ctx.rcx());
    let Some(q) = equeue::pin(handle) else { return crate::filesystem::sce_error(EBADF) };
    if let Err(e) = require_vnode_notes(notes) {
        return crate::filesystem::sce_error(e);
    }
    let e = vnode_descriptor_error(fd);
    if e != 0 {
        return crate::filesystem::sce_error(e);
    }
    let edited = q.edit_event(fd as u32 as u64, FILTER_FILE, |e| {
        e.event.udata = udata;
        e.filter.notes = notes;
    });
    if edited != ERR_NOT_FOUND {
        return edited;
    }
    match add_vnode_event(&q, fd, notes, EV_CLEAR, udata) {
        0 => OK,
        e => crate::filesystem::sce_error(e),
    }
}

/// sceKernelDeleteFileEvent(eq, int fd)
fn cb_delete_file_event(ctx: &mut CpuContext) -> HleResult {
    let (handle, fd) = (ctx.rdi(), ctx.rsi() as i32);
    let Some(q) = equeue::pin(handle) else { return crate::filesystem::sce_error(EBADF) };
    if fd < 0 || !crate::filesystem::descriptor_open(fd) {
        return crate::filesystem::sce_error(EBADF);
    }
    q.delete_event(fd as u32 as u64, FILTER_FILE)
}

pub(crate) static EXPORTS: &[Export] = &[
    ("nh2IFMgKTv8", "kqueue", "libKernel", cb_kqueue),
    ("RW-GEfpnsqg", "kevent", "libKernel", cb_kevent),
    ("cG3t15OK4Ow", "sceKernelAddFileEvent", "libKernel", cb_add_file_event),
    ("UmXngHKB6is", "sceKernelDeleteFileEvent", "libKernel", cb_delete_file_event),
];

#[cfg(all(test, unix))]
mod tests;
