// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Event queue behaviour: level/edge delivery, timers, duplicate adds,
//! delete hooks and queue lifetime under concurrency.

use super::*;
use crate::sync::test_util::vm_arena;
use std::sync::atomic::{AtomicBool, AtomicU32};
use std::sync::Weak;
use std::thread;

const ETIMEDOUT: HleResult = ERR_TIMED_OUT;
const EBADF: HleResult = ERR_BAD_QUEUE;
const ENOENT: HleResult = ERR_NOT_FOUND;

fn add_user(eq: u64, id: u64, edge: bool) -> HleResult {
    add_event(eq, user_event(id, if edge { EV_ADD | EV_CLEAR } else { EV_ADD }))
}

fn wait_n(eq: u64, cap: usize, timeout: Option<u32>) -> Result<Vec<KernelEvent>, HleResult> {
    wait(eq, cap, timeout)
}

#[test]
fn level_user_event_delivery() {
    let eq = create_named("level-user-events");
    assert_eq!(add_user(eq, 1, false), OK);
    assert_eq!(add_user(eq, 2, false), OK);
    assert_eq!(add_user(eq, 3, true), OK);
    for id in 1..=3 {
        assert_eq!(trigger(eq, id, FILTER_USER, id), OK);
    }
    let evs = wait_n(eq, 16, Some(0)).unwrap();
    assert_eq!(evs.len(), 3, "each level event once per wait");
    for id in 1..=3 {
        let e = evs.iter().find(|e| e.ident == id).unwrap();
        assert_eq!((e.data, e.udata), (id, id), "distinct payloads");
    }
    let evs = wait_n(eq, 16, Some(0)).unwrap();
    assert_eq!(evs.len(), 2, "levels persist while the edge clears");
    assert!(evs.iter().any(|e| e.ident == 1) && evs.iter().any(|e| e.ident == 2));
    assert_eq!(delete_event(eq, 1, FILTER_USER), OK);
    let evs = wait_n(eq, 16, Some(0)).unwrap();
    assert_eq!((evs.len(), evs[0].ident), (1, 2));
    assert_eq!(delete_event(eq, 2, FILTER_USER), OK);
    assert_eq!(wait_n(eq, 16, Some(0)), Err(ETIMEDOUT));
    assert_eq!(trigger(eq, 3, FILTER_USER, 0), OK);
    let evs = wait_n(eq, 16, Some(0)).unwrap();
    assert_eq!((evs.len(), evs[0].ident), (1, 3), "retriggered edge delivered once");
    assert_eq!(wait_n(eq, 16, Some(0)), Err(ETIMEDOUT));
    assert!(delete(eq));
}

#[test]
fn periodic_timer_events() {
    assert_eq!(add_timer(0, 10, 1_000_000, true, 0), EBADF);
    let eq = create_named("periodic-timer");
    assert_eq!(add_timer(eq, 10, 1_000_000, true, 0x1234), OK);
    thread::sleep(Duration::from_millis(5));
    let evs = wait_n(eq, 2, Some(0)).unwrap();
    assert_eq!(evs.len(), 1, "periodic occurrences aggregate into one event");
    let e = evs[0];
    assert!(e.ident == 10 && e.filter == FILTER_TIMER && e.data >= 5 && e.udata == 0x1234);
    let evs = wait_n(eq, 2, Some(100_000)).unwrap();
    assert!(evs.len() == 1 && evs[0].data >= 1, "periodic timer stays registered");

    thread::sleep(Duration::from_millis(5));
    // Re-adding updates the interval and user data but keeps what accrued.
    assert_eq!(add_timer(eq, 10, 3_000_000_000u64 * 1000, true, 0x5678), OK);
    let evs = wait_n(eq, 2, Some(0)).unwrap();
    assert!(evs.len() == 1 && evs[0].data >= 5 && evs[0].udata == 0x5678);
    assert_eq!(wait_n(eq, 2, Some(0)), Err(ETIMEDOUT), "update replaces the interval");
    assert_eq!(delete_event(eq, 10, FILTER_TIMER), OK);

    assert_eq!(add_timer(eq, 10, 0, true, 0x5678), OK);
    for _ in 0..2 {
        assert_eq!(add_timer(eq, 10, 0, true, 0x5678), OK);
    }
    for _ in 0..2 {
        let evs = wait_n(eq, 2, Some(0)).unwrap();
        assert!(evs.len() == 1 && evs[0].data == 1, "zero interval stays ready");
    }
    assert_eq!(delete_event(eq, 10, FILTER_TIMER), OK);

    assert_eq!(add_timer(eq, 10, 1_000_000, false, 0x5678), OK);
    let evs = wait_n(eq, 2, Some(100_000)).unwrap();
    assert!(evs.len() == 1 && evs[0].filter == FILTER_HR_TIMER && evs[0].udata == 0x5678);
    assert_eq!(delete_event(eq, 10, FILTER_HR_TIMER), ENOENT, "HR timer removed after one delivery");
    assert!(delete(eq));
}

struct DuplicateOwner {
    delete_count: AtomicU32,
}

fn queue_duplicate(e: &mut EqueueEvent, data: u64) {
    let next = KernelEvent { data, ..e.event };
    if e.triggered {
        e.pending.push_back(next);
    } else {
        e.event = next;
        e.triggered = true;
    }
}

fn reset_duplicate(e: &mut EqueueEvent) {
    e.triggered = false;
    e.event.data = 0;
}

fn poison_duplicate(_: &mut EqueueEvent, _: u64) {
    panic!("duplicate add replaced the trigger hook");
}

#[test]
fn duplicate_add_preserves_event_state() {
    let eq = create_named("duplicate-add");
    let original = Arc::new(DuplicateOwner { delete_count: AtomicU32::new(0) });
    let weak_original = Arc::downgrade(&original);
    let o = original.clone();
    let event = EqueueEvent {
        event: KernelEvent { ident: 17, filter: FILTER_VIDEO_OUT, udata: 0x1111, ..Default::default() },
        filter: Filter {
            trigger: Some(queue_duplicate),
            reset: Some(reset_duplicate),
            on_delete: Some(Arc::new(move |_, _| {
                o.delete_count.fetch_add(1, Ordering::Relaxed);
            })),
            ..Default::default()
        },
        ..Default::default()
    };
    assert_eq!(add_event(eq, event), OK);
    assert_eq!(trigger(eq, 17, FILTER_VIDEO_OUT, 0x1234), OK);
    assert_eq!(trigger(eq, 17, FILTER_VIDEO_OUT, 0x5678), OK);

    let replacement = Arc::new(DuplicateOwner { delete_count: AtomicU32::new(0) });
    let weak_replacement = Arc::downgrade(&replacement);
    let r = replacement.clone();
    let duplicate = EqueueEvent {
        deadline_ns: 1,
        event: KernelEvent { ident: 17, filter: FILTER_VIDEO_OUT, data: 0x7fff_ffff, udata: 0x2222, ..Default::default() },
        filter: Filter {
            trigger: Some(poison_duplicate),
            on_delete: Some(Arc::new(move |_, _| {
                r.delete_count.fetch_add(1, Ordering::Relaxed);
            })),
            ..Default::default()
        },
        ..Default::default()
    };
    assert_eq!(add_event(eq, duplicate), OK);
    drop(replacement);
    assert!(weak_replacement.upgrade().is_none(), "duplicate owner is not retained");
    drop(original);
    assert!(weak_original.upgrade().is_some(), "original owner stays retained");

    let evs = wait_n(eq, 2, Some(0)).unwrap();
    assert_eq!(evs.len(), 2, "pending count preserved");
    assert_eq!((evs[0].data, evs[1].data), (0x1234, 0x5678));
    assert_eq!((evs[0].udata, evs[1].udata), (0x2222, 0x2222), "user data updated on current and pending");

    let evs = wait_n(eq, 1, Some(0)).unwrap();
    assert_eq!(evs.len(), 1, "duplicate add updated the deadline");
    assert_eq!((evs[0].data, evs[0].udata), (0, 0x2222));
    for v in [0x9abc, 0xdef0] {
        assert_eq!(trigger(eq, 17, FILTER_VIDEO_OUT, v), OK);
    }
    for v in [0x9abc, 0xdef0] {
        let evs = wait_n(eq, 1, Some(0)).unwrap();
        assert_eq!((evs.len(), evs[0].data), (1, v), "capacity-one reads keep order");
    }
    let retained = weak_original.upgrade().unwrap();
    assert_eq!(delete_event(eq, 17, FILTER_VIDEO_OUT), OK);
    assert_eq!(retained.delete_count.load(Ordering::Relaxed), 1, "original delete hook kept");
    drop(retained);
    assert!(weak_original.upgrade().is_none(), "owner released on delete");
    assert!(delete(eq));
}

#[test]
fn delete_hook_state_outlives_its_source() {
    let eq = create_named("shared-port-state");
    struct PortState {
        registrations: Mutex<Vec<Arc<u64>>>,
    }
    let stage = Arc::new(AtomicU32::new(0));
    let port = Arc::new(PortState { registrations: Mutex::new(Vec::new()) });
    let weak_port = Arc::downgrade(&port);
    let registration = Arc::new(0x1234_5678_9abc_def0u64);
    let weak_registration = Arc::downgrade(&registration);
    port.registrations.lock().unwrap().push(registration.clone());
    let (st, p, reg) = (stage.clone(), port.clone(), registration.clone());
    let event = EqueueEvent {
        event: KernelEvent { ident: 8, filter: FILTER_VIDEO_OUT, ..Default::default() },
        filter: Filter {
            on_delete: Some(Arc::new(move |_, _| {
                st.store(1, Ordering::Release);
                while st.load(Ordering::Acquire) != 2 {
                    thread::yield_now();
                }
                p.registrations.lock().unwrap().retain(|r| !Arc::ptr_eq(r, &reg));
                assert_eq!(*reg, 0x1234_5678_9abc_def0, "callback state survives the port");
            })),
            ..Default::default()
        },
        ..Default::default()
    };
    assert_eq!(add_event(eq, event), OK);
    let closer = thread::spawn(move || delete(eq));
    while stage.load(Ordering::Acquire) != 1 {
        thread::yield_now();
    }
    let detached = std::mem::take(&mut *port.registrations.lock().unwrap());
    drop(registration);
    drop(detached);
    drop(port);
    assert!(weak_port.upgrade().is_some() && weak_registration.upgrade().is_some());
    stage.store(2, Ordering::Release);
    assert!(closer.join().unwrap());
    assert!(weak_registration.upgrade().is_none() && weak_port.upgrade().is_none(), "released after close");
}

#[test]
fn delete_hook_runs_after_registry_removal_and_owns_payload() {
    let eq = create_named("owned-callback");
    let stage = Arc::new(AtomicU32::new(0));
    let count = Arc::new(AtomicU32::new(0));
    let payload = Arc::new(0xc0de_c0de_c0de_c0deu64);
    let weak_payload: Weak<u64> = Arc::downgrade(&payload);
    let (st, c, pl) = (stage.clone(), count.clone(), payload.clone());
    let event = EqueueEvent {
        event: KernelEvent { ident: 2, filter: FILTER_VIDEO_OUT, ..Default::default() },
        filter: Filter {
            on_delete: Some(Arc::new(move |q, _| {
                assert!(pin(q).is_none(), "hook runs after registry removal");
                c.fetch_add(1, Ordering::Relaxed);
                st.store(1, Ordering::Release);
                while st.load(Ordering::Acquire) != 2 {
                    thread::yield_now();
                }
                assert_eq!(*pl, 0xc0de_c0de_c0de_c0de);
            })),
            ..Default::default()
        },
        ..Default::default()
    };
    assert_eq!(add_event(eq, event), OK);
    let closer = thread::spawn(move || delete(eq));
    while stage.load(Ordering::Acquire) != 1 {
        thread::yield_now();
    }
    assert!(pin(eq).is_none());
    drop(payload);
    assert!(weak_payload.upgrade().is_some(), "hook keeps its payload");
    stage.store(2, Ordering::Release);
    assert!(closer.join().unwrap());
    assert_eq!(count.load(Ordering::Relaxed), 1);
    assert!(weak_payload.upgrade().is_none(), "payload released with the event");
}

fn counting_event(ident: u64, filter: i16, count: &Arc<AtomicU32>) -> EqueueEvent {
    let c = count.clone();
    EqueueEvent {
        event: KernelEvent { ident, filter, ..Default::default() },
        filter: Filter {
            on_delete: Some(Arc::new(move |_, _| {
                c.fetch_add(1, Ordering::Relaxed);
            })),
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn pinned_close() {
    let eq = create_named("pinned-close");
    let count = Arc::new(AtomicU32::new(0));
    assert_eq!(add_event(eq, counting_event(1, FILTER_VIDEO_OUT, &count)), OK);
    let owner = pin(eq).expect("pin live queue");
    assert!(delete(eq));
    assert_eq!(count.load(Ordering::Relaxed), 1, "close invokes the hook once");
    assert!(pin(eq).is_none());
    assert_eq!(trigger(eq, 1, FILTER_VIDEO_OUT, 0), EBADF, "stale trigger rejected");
    assert!(!delete(eq), "second delete rejected");
    // Operations on a pinned but closed queue fail too.
    assert_eq!(owner.trigger_event(1, FILTER_VIDEO_OUT, 0), EBADF);
    drop(owner);
    assert_eq!(count.load(Ordering::Relaxed), 1, "dropping the last pin does not repeat the hook");
}

#[test]
fn stale_handle_never_aliases_new_queue() {
    let stale = create_named("stale-handle");
    assert!(delete(stale));
    let replacement = create_named("replacement");
    assert_ne!(stale, replacement, "handles are never recycled");
    assert!(pin(stale).is_none());
    assert_eq!(add_user(stale, 11, false), EBADF);
    assert_eq!(add_user(replacement, 11, false), OK);
    assert_eq!(trigger(stale, 11, FILTER_USER, 0), EBADF);
    assert_eq!(trigger(replacement, 11, FILTER_USER, 0), OK);
    assert!(delete(replacement));
}

fn concurrent_ok(r: HleResult) {
    assert!(r == OK || r == EBADF || r == ENOENT, "unexpected result {r:#x}");
}

#[test]
fn concurrent_close_runs_hook_once() {
    for _ in 0..64 {
        let eq = create_named("callback-race");
        let count = Arc::new(AtomicU32::new(0));
        assert_eq!(add_event(eq, counting_event(9, FILTER_GRAPHICS, &count)), OK);
        let start = Arc::new(AtomicBool::new(false));
        let s = start.clone();
        let t = thread::spawn(move || {
            while !s.load(Ordering::Acquire) {
                thread::yield_now();
            }
            for _ in 0..256 {
                concurrent_ok(trigger(eq, 9, FILTER_GRAPHICS, 0));
            }
        });
        start.store(true, Ordering::Release);
        assert!(delete(eq));
        t.join().unwrap();
        assert_eq!(count.load(Ordering::Relaxed), 1);
    }
}

#[test]
fn concurrent_delete() {
    for _ in 0..64 {
        let eq = create_named("concurrent-delete");
        let ev = EqueueEvent { event: KernelEvent { ident: 7, filter: FILTER_USER, ..Default::default() }, ..Default::default() };
        assert_eq!(add_event(eq, ev.clone()), OK);
        let start = Arc::new(AtomicBool::new(false));
        let (s1, s2) = (start.clone(), start.clone());
        let mutate = thread::spawn(move || {
            while !s1.load(Ordering::Acquire) {
                thread::yield_now();
            }
            for _ in 0..64 {
                concurrent_ok(add_event(eq, ev.clone()));
                concurrent_ok(trigger(eq, 7, FILTER_USER, 0));
                concurrent_ok(delete_event(eq, 7, FILTER_USER));
            }
        });
        let trig = thread::spawn(move || {
            while !s2.load(Ordering::Acquire) {
                thread::yield_now();
            }
            for _ in 0..128 {
                concurrent_ok(trigger(eq, 7, FILTER_USER, 0));
            }
        });
        start.store(true, Ordering::Release);
        assert!(delete(eq));
        mutate.join().unwrap();
        trig.join().unwrap();
        assert!(pin(eq).is_none());
    }
}

#[test]
fn wait_blocks_until_trigger_and_delete_wakes_waiter() {
    let eq = create();
    assert_eq!(add_user(eq, 7, true), OK);
    let t = thread::spawn(move || {
        thread::sleep(Duration::from_millis(30));
        trigger(eq, 7, FILTER_USER, 5)
    });
    let start = Instant::now();
    let evs = wait_n(eq, 1, None).unwrap();
    assert!(start.elapsed() >= Duration::from_millis(20));
    assert_eq!((evs[0].ident, evs[0].udata), (7, 5));
    assert_eq!(t.join().unwrap(), OK);

    let start = Instant::now();
    assert_eq!(wait_n(eq, 1, Some(20_000)), Err(ETIMEDOUT));
    assert!(start.elapsed() >= Duration::from_millis(20));
    let t = thread::spawn(move || wait_n(eq, 1, None));
    thread::sleep(Duration::from_millis(20));
    assert!(delete(eq));
    assert_eq!(t.join().unwrap(), Err(EBADF));
    assert_eq!(wait_n(eq, 1, None), Err(EBADF));
}

#[test]
fn display_events_coalesce_count() {
    let eq = create();
    for _ in 0..3 {
        trigger_display_event(eq, VIDEO_OUT_EVENT_FLIP, FILTER_VIDEO_OUT, 0x1234_0000, 9);
    }
    let evs = wait_n(eq, 4, Some(0)).unwrap();
    assert_eq!(evs.len(), 1);
    assert_eq!((evs[0].data >> 12) & 0xF, 3);
    assert_eq!(evs[0].data & !0xFFFF, 0x1234_0000);
    assert_eq!(evs[0].udata, 9);
    assert_eq!(pending_count(eq), 0);
    delete(eq);
}

#[test]
fn graphics_interrupts_queue_one_occurrence_per_trigger() {
    let eq = create();
    assert!(register_event(eq, 0x40, FILTER_GRAPHICS, 0x99, EV_CLEAR, 0));
    assert_eq!(trigger_registered_events_by_filter(FILTER_GRAPHICS, 1, 0x40), 1);
    assert_eq!(trigger_registered_events_by_filter(FILTER_GRAPHICS, 2, 0x40), 1);
    assert_eq!(pending_count(eq), 2);
    let evs = wait_n(eq, 8, Some(0)).unwrap();
    assert_eq!(evs.len(), 2);
    assert_eq!((evs[0].data, evs[0].fflags, evs[0].udata), (1, 1, 0x99));
    assert_eq!((evs[1].data, evs[1].fflags), (2, 2));
    assert_eq!(pending_count(eq), 0);
    delete(eq);
}

fn readiness(fd: i32, _lw: u64) -> Option<ReadReadiness> {
    (fd == 4242).then_some(ReadReadiness { ready: true, available: 17, flags: 0 })
}

#[test]
fn guest_callbacks() {
    const BASE: u64 = 0x5CF1_0000_0000;
    let vm = vm_arena(BASE);
    let mut ctx = CpuContext::new(vm);
    ctx.write_bytes(BASE + 0x40, b"q\0");
    ctx.set(crate::CpuRegister::Rdi, BASE);
    ctx.set(crate::CpuRegister::Rsi, 0);
    assert_eq!(cb_create_equeue(&mut ctx), ERR_INVALID_ARGUMENT, "name required");
    ctx.set(crate::CpuRegister::Rsi, BASE + 0x40);
    assert_eq!(cb_create_equeue(&mut ctx), OK);
    let eq = ctx.read_u64(BASE).unwrap();
    add_read_readiness_provider(readiness);
    ctx.regs = [0; 16];
    ctx.set(crate::CpuRegister::Rdi, eq);
    ctx.set(crate::CpuRegister::Rsi, 4242);
    ctx.set(crate::CpuRegister::Rdx, 1);
    ctx.set(crate::CpuRegister::Rcx, 0x55);
    assert_eq!(cb_add_read_event(&mut ctx), OK);
    ctx.set(crate::CpuRegister::Rsi, 4243);
    assert_eq!(cb_add_read_event(&mut ctx), ERR_NOT_FOUND);
    ctx.write_u32(BASE + 0x10, 0);
    ctx.set(crate::CpuRegister::Rsi, BASE + 0x100);
    ctx.set(crate::CpuRegister::Rdx, 0);
    ctx.set(crate::CpuRegister::Rcx, BASE + 0x8);
    ctx.set(crate::CpuRegister::R8, BASE + 0x10);
    assert_eq!(cb_wait_equeue(&mut ctx), ERR_INVALID_ARGUMENT, "num < 1");
    ctx.set(crate::CpuRegister::Rdx, 4);
    assert_eq!(cb_wait_equeue(&mut ctx), OK);
    assert_eq!(ctx.read_u32(BASE + 0x8), Some(1));
    assert_eq!(ctx.read_u64(BASE + 0x100), Some(4242));
    ctx.set(crate::CpuRegister::Rdi, BASE + 0x100);
    assert_eq!(cb_get_event_filter(&mut ctx), FILTER_READ as i32 as u32 as i64);
    assert_eq!(cb_get_event_data(&mut ctx), 17);
    assert_eq!(cb_get_event_user_data(&mut ctx), 0x55);
    assert_eq!(cb_get_event_id(&mut ctx), 4242);
    assert_eq!(cb_get_event_fflags(&mut ctx), 0);
    assert_eq!(cb_get_event_error(&mut ctx), 0);
    ctx.write_bytes(BASE + 0x10A, &EV_ERROR.to_le_bytes());
    assert_eq!(cb_get_event_error(&mut ctx), 17);
    ctx.set(crate::CpuRegister::Rdi, 0);
    assert_eq!(cb_get_event_data(&mut ctx), 0);
    ctx.set(crate::CpuRegister::Rdi, eq);
    ctx.set(crate::CpuRegister::Rsi, 0);
    assert_eq!(cb_wait_equeue(&mut ctx), ERR_MEMORY_FAULT, "null event array");
    assert_eq!(cb_delete_equeue(&mut ctx), OK);
    assert_eq!(cb_delete_equeue(&mut ctx), EBADF);
    // AMPR adds/deletes succeed even without a queue.
    ctx.set(crate::CpuRegister::Rdi, 0);
    assert_eq!(cb_add_ampr_event(&mut ctx), OK);
    assert_eq!(cb_delete_ampr_event(&mut ctx), OK);
}

/// sceKernelAddReadEvent takes the read end of a pipe (its low-water mark
/// does not apply) and still refuses the write end.
#[test]
fn read_event_on_a_pipe() {
    use crate::filesystem::test_hooks as fs;
    const BASE: u64 = 0x5CF9_0000_0000;
    let vm = vm_arena(BASE);
    let mut ctx = CpuContext::new(vm);
    let eq = create_named("pipe");
    let (pr, pw) = fs::pipe(&ctx, BASE + 0x40);
    ctx.set(crate::CpuRegister::Rdi, eq);
    ctx.set(crate::CpuRegister::Rsi, pw as u64);
    ctx.set(crate::CpuRegister::Rdx, 1);
    assert_eq!(cb_add_read_event(&mut ctx), ERR_NOT_FOUND);
    ctx.set(crate::CpuRegister::Rsi, pr as u64);
    ctx.set(crate::CpuRegister::Rdx, 64);
    ctx.set(crate::CpuRegister::Rcx, 0x77);
    assert_eq!(cb_add_read_event(&mut ctx), OK);
    ctx.write_bytes(BASE + 0x80, b"x");
    assert_eq!(fs::write(&ctx, pw, BASE + 0x80, 1), Ok(1));
    ctx.set(crate::CpuRegister::Rsi, BASE + 0x100);
    ctx.set(crate::CpuRegister::Rdx, 4);
    ctx.set(crate::CpuRegister::Rcx, BASE + 0x8);
    ctx.set(crate::CpuRegister::R8, 0);
    assert_eq!(cb_wait_equeue(&mut ctx), OK);
    assert_eq!(ctx.read_u32(BASE + 0x8), Some(1));
    assert_eq!(ctx.read_u64(BASE + 0x100), Some(pr as u64));
    ctx.set(crate::CpuRegister::Rdi, BASE + 0x100);
    assert_eq!(cb_get_event_data(&mut ctx), 1);
    assert_eq!(cb_get_event_user_data(&mut ctx), 0x77);
    fs::close(pr).unwrap();
    fs::close(pw).unwrap();
}
