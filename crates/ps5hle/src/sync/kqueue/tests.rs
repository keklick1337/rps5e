// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! kqueue / kevent / vnode watches, after the reference's GuestKqueue
//! program (sockets share the descriptor space here, so an unknown
//! descriptor stands in for its first socket number).

use super::*;
use crate::context::CpuRegister;
use crate::filesystem::test_hooks as fs;
use ps5vm::{MemProtect, VirtualMemory};
use std::path::{Path, PathBuf};

const EVFILT_READ: i16 = -1;
const EVFILT_WRITE: i16 = -2;
const EVFILT_VNODE: i16 = -4;
const EVFILT_PROC: i16 = -5;
const EVFILT_TIMER: i16 = -7;
const EVFILT_USER: i16 = -11;
const EV_EOF: u16 = 0x8000;
const EV_DISPATCH: u16 = 0x0080;

struct Kq {
    ctx: CpuContext<'static>,
    errno: u64,
    base: u64,
}

const CHANGES: u64 = 0x100;
const EVENTS: u64 = 0x400;
const TIMEOUT: u64 = 0x80;
const SCRATCH: u64 = 0x800;

fn change(ident: u64, filter: i16, flags: u16, fflags: u32, data: i64, udata: u64) -> KernelEvent {
    KernelEvent { ident, filter, flags, fflags, data: data as u64, udata }
}

impl Kq {
    fn new(base: u64) -> Self {
        let vm: &'static VirtualMemory = Box::leak(Box::new(VirtualMemory::new()));
        vm.map(base, 0x2000, MemProtect::RW).unwrap();
        let ctx = CpuContext::new(vm);
        let errno = crate::kernel::bind_test_errno(&ctx);
        Kq { ctx, errno, base }
    }

    fn call(&mut self, cb: crate::registry::HleCallback, args: &[u64]) -> HleResult {
        let regs = [CpuRegister::Rdi, CpuRegister::Rsi, CpuRegister::Rdx, CpuRegister::Rcx, CpuRegister::R8, CpuRegister::R9];
        for (r, v) in regs.iter().zip(args) {
            self.ctx.set(*r, *v);
        }
        cb(&mut self.ctx)
    }

    fn errno(&self) -> i32 {
        self.ctx.read_u32(self.errno).unwrap() as i32
    }

    fn kqueue(&mut self) -> i32 {
        let fd = self.call(cb_kqueue, &[]) as i32;
        assert!(fd > 2);
        fd
    }

    fn kevent(&mut self, kq: i32, changes: &[KernelEvent], nevents: i32, timeout: Option<(i64, i64)>) -> (HleResult, Vec<KernelEvent>) {
        for (i, c) in changes.iter().enumerate() {
            self.ctx.write_bytes(self.base + CHANGES + i as u64 * 0x20, &c.to_bytes());
        }
        let ts = match timeout {
            Some((s, ns)) => {
                self.ctx.write_u64(self.base + TIMEOUT, s as u64);
                self.ctx.write_u64(self.base + TIMEOUT + 8, ns as u64);
                self.base + TIMEOUT
            }
            None => 0,
        };
        let list = if nevents > 0 { self.base + EVENTS } else { 0 };
        let cl = if changes.is_empty() { 0 } else { self.base + CHANGES };
        let r = self.call(cb_kevent, &[kq as u64, cl, changes.len() as u64, list, nevents as u32 as u64, ts]);
        let n = if r > 0 { r as usize } else { 0 };
        let evs = (0..n.min(nevents.max(0) as usize))
            .map(|i| event_from_bytes(&self.ctx.read_bytes(self.base + EVENTS + i as u64 * 0x20, 0x20).unwrap()))
            .collect();
        (r, evs)
    }

    fn apply(&mut self, kq: i32, c: KernelEvent) -> HleResult {
        self.kevent(kq, &[c], 0, None).0
    }

    fn poll(&mut self, kq: i32) -> Vec<KernelEvent> {
        let (r, evs) = self.kevent(kq, &[], 4, Some((0, 0)));
        assert!(r >= 0, "poll failed: errno {}", self.errno());
        evs
    }

    fn wait(&mut self, kq: i32) -> Vec<KernelEvent> {
        let (r, evs) = self.kevent(kq, &[], 4, Some((1, 0)));
        assert!(r >= 0, "wait failed: errno {}", self.errno());
        evs
    }

    fn require_failure(&self, r: HleResult, errno: i32) {
        assert_eq!(r, -1);
        assert_eq!(self.errno(), errno);
    }

    fn write(&mut self, fd: i32, bytes: &[u8]) {
        self.ctx.write_bytes(self.base + SCRATCH, bytes);
        assert_eq!(fs::write(&self.ctx, fd, self.base + SCRATCH, bytes.len() as u64), Ok(bytes.len() as i64));
    }

    fn read(&mut self, fd: i32, len: u64) -> i64 {
        fs::read(&self.ctx, fd, self.base + SCRATCH, len).unwrap()
    }

    fn socketpair(&mut self, ty: i32) -> (i32, i32) {
        crate::net::test_socketpair(&self.ctx, ty, self.base + 0x60)
    }
}

fn send(fd: i32, bytes: &[u8]) {
    assert_eq!(crate::net::kernel_socket_send_bytes(fd, bytes).unwrap(), Ok(bytes.len() as i64));
}

fn recv(fd: i32, len: usize) -> usize {
    crate::net::kernel_socket_recv_bytes(fd, len).unwrap().unwrap().len()
}

fn close(fd: i32) {
    assert_eq!(fs::close(fd), Ok(0));
}

/// A fresh scratch path under the repository's local/tmp/ directory.
fn temp_path(tag: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../local/tmp/kqueue-tests");
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(format!("{tag}-{}", std::process::id()))
}

fn settle() {
    std::thread::sleep(Duration::from_millis(200));
}

/// EVFILT_READ on standard input (descriptor 0) is accepted and never
/// fires.
#[test]
fn read_event_on_standard_input() {
    let mut k = Kq::new(0x7E5E_0000_0000);
    let kq = k.kqueue();
    assert_eq!(k.apply(kq, change(0, EVFILT_READ, EV_ADD, 0, 0, 7)), 0);
    assert!(k.poll(kq).is_empty());
    close(kq);
}

#[test]
fn create_and_close() {
    let mut k = Kq::new(0x7E51_0000_0000);
    let kq = k.kqueue();
    let other = k.kqueue();
    assert_ne!(kq, other);
    assert!(k.poll(kq).is_empty());
    close(kq);
    let r = k.kevent(kq, &[], 4, Some((0, 0))).0;
    k.require_failure(r, EBADF);
    close(other);
    let (r0, w0) = fs::pipe(&k.ctx, k.base + 0x40);
    let r = k.kevent(r0, &[], 4, Some((0, 0))).0;
    k.require_failure(r, EBADF);
    close(r0);
    close(w0);
}

#[test]
fn timers() {
    let mut k = Kq::new(0x7E52_0000_0000);
    let kq = k.kqueue();
    let marker = 0x1234;
    assert_eq!(k.apply(kq, change(1, EVFILT_TIMER, EV_ADD, 0, 10, marker)), 0);
    assert!(k.poll(kq).is_empty());
    let ev = k.wait(kq);
    assert_eq!(ev.len(), 1);
    assert!(ev[0].ident == 1 && ev[0].filter == EVFILT_TIMER && ev[0].data >= 1 && ev[0].udata == marker);
    assert!(ev[0].flags & EV_CLEAR != 0);
    assert_eq!(k.wait(kq).len(), 1);
    assert_eq!(k.apply(kq, change(1, EVFILT_TIMER, EV_DELETE, 0, 0, 0)), 0);
    let r = k.apply(kq, change(1, EVFILT_TIMER, EV_DELETE, 0, 0, 0));
    k.require_failure(r, ENOENT);
    assert_eq!(k.apply(kq, change(2, EVFILT_TIMER, EV_ADD | EV_ONESHOT, NOTE_USECONDS, 0, 0)), 0);
    let ev = k.wait(kq);
    assert!(ev.len() == 1 && ev[0].ident == 2 && ev[0].data == 1);
    assert!(k.poll(kq).is_empty());
    let r = k.apply(kq, change(2, EVFILT_TIMER, EV_DELETE, 0, 0, 0));
    k.require_failure(r, ENOENT);
    assert_eq!(k.apply(kq, change(5, EVFILT_TIMER, EV_ADD | EV_ONESHOT, NOTE_NSECONDS, 150_000_001, 0)), 0);
    let (n, ev) = k.kevent(kq, &[], 4, Some((2, 0)));
    assert!(n == 1 && ev[0].ident == 5 && ev[0].data == 1);
    let r = k.apply(kq, change(3, EVFILT_TIMER, EV_ADD, 0, -1, 0));
    k.require_failure(r, EINVAL);
    let r = k.apply(kq, change(3, EVFILT_TIMER, EV_ADD, 0x10, 5, 0));
    k.require_failure(r, EINVAL);
    let r = k.apply(kq, change(3, EVFILT_TIMER, EV_ADD, NOTE_USECONDS | 1, 5, 0));
    k.require_failure(r, EINVAL);
    assert_eq!(k.apply(kq, change(4, EVFILT_TIMER, EV_ADD, 0, 100_000, 0)), 0);
    // Re-adding a live timer is allowed: the expired periods fire first
    // and the new interval takes over (measured on hardware).
    assert_eq!(k.apply(kq, change(4, EVFILT_TIMER, EV_ADD, 0, 5, 0)), 0);
    close(kq);
}

/// Disabling a timer keeps its phase: after EV_ENABLE the first delivery
/// sums the expiries missed while disabled and the next keeps the
/// cadence; a change without a new period never re-arms the timer.
#[test]
fn timer_disable_then_enable_sums_missed_expiries() {
    let mut k = Kq::new(0x7E77_0000_0000);
    let kq = k.kqueue();
    let marker = 0x5555;
    assert_eq!(k.apply(kq, change(1, EVFILT_TIMER, EV_ADD, NOTE_USECONDS, 50_000, marker)), 0);
    assert_eq!(k.apply(kq, change(1, EVFILT_TIMER, EV_DISABLE, 0, 0, marker)), 0);
    std::thread::sleep(std::time::Duration::from_millis(170));
    assert!(k.poll(kq).is_empty());
    assert_eq!(k.apply(kq, change(1, EVFILT_TIMER, EV_ENABLE, 0, 0, marker)), 0);
    let ev = k.wait(kq);
    assert_eq!(ev.len(), 1);
    assert_eq!(ev[0].udata, marker);
    assert!((2..=6).contains(&ev[0].data), "missed expiries summed: {}", ev[0].data);
    let ev = k.wait(kq);
    assert_eq!(ev.len(), 1);
    assert!((1..=2).contains(&ev[0].data), "phase kept: {}", ev[0].data);
    // EV_ADD without a new period keeps the interval: an enable, not a
    // re-arm to the default one.
    assert_eq!(k.apply(kq, change(1, EVFILT_TIMER, EV_ADD | EV_ENABLE, 0, 0, marker)), 0);
    // A period change without EV_ADD is refused.
    let r = k.apply(kq, change(1, EVFILT_TIMER, 0, 0, 5, 0));
    k.require_failure(r, EINVAL);
    assert_eq!(k.apply(kq, change(1, EVFILT_TIMER, EV_DELETE, 0, 0, 0)), 0);
    close(kq);
}

#[test]
fn user_events() {
    let mut k = Kq::new(0x7E53_0000_0000);
    let kq = k.kqueue();
    let marker = 0x5555;
    assert_eq!(k.apply(kq, change(5, EVFILT_USER, EV_ADD | EV_CLEAR, 0, 0, marker)), 0);
    assert!(k.poll(kq).is_empty());
    let handle = kq_handle(kq).unwrap();
    let trigger = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(20));
        let q = equeue::pin(handle).unwrap();
        let mut c = change(5, EVFILT_USER, 0, NOTE_TRIGGER | NOTE_FFCOPY | 0x123, 77, marker);
        assert_eq!(register(&q, kq, &mut c), 0);
    });
    let (n, ev) = k.kevent(kq, &[], 4, None);
    trigger.join().unwrap();
    assert_eq!(n, 1);
    assert!(ev[0].ident == 5 && ev[0].filter == EVFILT_USER && ev[0].fflags == 0x123 && ev[0].data == 77);
    assert!(ev[0].udata == marker && ev[0].flags == EV_CLEAR);
    assert!(k.poll(kq).is_empty());
    assert_eq!(k.apply(kq, change(5, EVFILT_USER, 0, NOTE_TRIGGER | NOTE_FFOR | 0x400, 78, 0)), 0);
    let ev = k.poll(kq);
    assert!(ev.len() == 1 && ev[0].fflags == 0x523 && ev[0].data == 78);
    assert_eq!(k.apply(kq, change(6, EVFILT_USER, EV_ADD, 0, 0, 0)), 0);
    assert_eq!(k.apply(kq, change(6, EVFILT_USER, 0, NOTE_TRIGGER, 0, 0)), 0);
    for _ in 0..2 {
        let ev = k.poll(kq);
        assert!(ev.len() == 1 && ev[0].ident == 6);
    }
    assert_eq!(k.apply(kq, change(6, EVFILT_USER, EV_CLEAR, 0, 0, 0)), 0);
    assert!(k.poll(kq).is_empty());
    assert_eq!(k.apply(kq, change(6, EVFILT_USER, EV_DELETE, 0, 0, 0)), 0);
    let (n, receipt) = k.kevent(kq, &[change(7, EVFILT_USER, EV_ADD | EV_RECEIPT, 0x55, 9, marker)], 2, None);
    assert_eq!(n, 1);
    assert!(receipt[0].ident == 7 && receipt[0].flags == EV_ERROR && receipt[0].data == 0 && receipt[0].fflags == 0);
    assert_eq!(k.apply(kq, change(7, EVFILT_USER, 0, NOTE_TRIGGER, 0, 0)), 0);
    let ev = k.poll(kq);
    assert!(ev.len() == 1 && ev[0].fflags == 0x55 && ev[0].data == 0 && ev[0].flags == EV_RECEIPT);
    close(kq);
}

#[test]
fn pipe_read() {
    let mut k = Kq::new(0x7E54_0000_0000);
    let kq = k.kqueue();
    let (r, w) = fs::pipe(&k.ctx, k.base + 0x40);
    let marker = 0x77;
    assert_eq!(k.apply(kq, change(r as u64, EVFILT_READ, EV_ADD, 0, 0, marker)), 0);
    assert!(k.poll(kq).is_empty());
    k.write(w, b"abc");
    let ev = k.poll(kq);
    assert!(ev.len() == 1 && ev[0].ident == r as u64 && ev[0].filter == EVFILT_READ && ev[0].data == 3);
    assert!(ev[0].udata == marker && ev[0].flags & EV_EOF == 0);
    assert_eq!(k.read(r, 4), 3);
    assert!(k.poll(kq).is_empty());
    let vm = k.ctx.vm;
    let writer = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(50));
        let ctx = CpuContext::new(vm);
        ctx.write_bytes(0x7E54_0000_0000 + 0x1800, b"d");
        assert_eq!(fs::write(&ctx, w, 0x7E54_0000_0000 + 0x1800, 1), Ok(1));
    });
    let start = Instant::now();
    let (n, ev) = k.kevent(kq, &[], 4, Some((5, 0)));
    let waited = start.elapsed();
    writer.join().unwrap();
    assert_eq!(n, 1);
    assert!(waited >= Duration::from_millis(40) && waited < Duration::from_secs(4));
    assert!(ev[0].ident == r as u64 && ev[0].data == 1);
    close(w);
    let ev = k.poll(kq);
    assert!(ev.len() == 1 && ev[0].data == 1 && ev[0].flags & EV_EOF != 0);
    assert_eq!(k.read(r, 4), 1);
    let ev = k.poll(kq);
    assert!(ev.len() == 1 && ev[0].data == 0 && ev[0].flags & EV_EOF != 0);
    close(r);
    close(kq);
}

#[test]
fn socket_pair() {
    let mut k = Kq::new(0x7E55_0000_0000);
    let kq = k.kqueue();
    let (a, b) = k.socketpair(1);
    let marker = 0x99;
    assert_eq!(k.apply(kq, change(a as u64, EVFILT_READ, EV_ADD, 0, 0, marker)), 0);
    assert!(k.poll(kq).is_empty());
    send(b, b"abc");
    let ev = k.wait(kq);
    assert!(ev.len() == 1 && ev[0].ident == a as u64 && ev[0].filter == EVFILT_READ && ev[0].data == 3 && ev[0].udata == marker);
    assert_eq!(k.poll(kq).len(), 1);
    assert_eq!(recv(a, 4), 3);
    assert!(k.poll(kq).is_empty());
    assert_eq!(k.apply(kq, change(b as u64, EVFILT_WRITE, EV_ADD | EV_ONESHOT, 0, 0, 0)), 0);
    let ev = k.wait(kq);
    assert!(ev.len() == 1 && ev[0].filter == EVFILT_WRITE);
    let r = k.apply(kq, change(b as u64, EVFILT_WRITE, EV_DELETE, 0, 0, 0));
    k.require_failure(r, ENOENT);
    let r = k.apply(kq, change(b as u64, EVFILT_READ, EV_ADD | EV_CLEAR, 0, 0, 0));
    k.require_failure(r, EINVAL);
    // Write events report the free send space.
    let space = k.kqueue();
    assert_eq!(k.apply(space, change(b as u64, EVFILT_WRITE, EV_ADD, 0, 0, 0)), 0);
    let ev = k.poll(space);
    assert!(ev.len() == 1 && ev[0].filter == EVFILT_WRITE);
    let idle = ev[0].data as i64;
    assert!(idle > 0);
    send(b, &[b'p'; 1000]);
    let ev = k.poll(space);
    assert!(ev.len() == 1 && (ev[0].data as i64) <= idle - 1000);
    assert_eq!(recv(a, 1000), 1000);
    let ev = k.poll(space);
    assert!(ev.len() == 1 && ev[0].data as i64 == idle);
    assert_eq!(k.apply(space, change(b as u64, EVFILT_WRITE, EV_DELETE, 0, 0, 0)), 0);
    close(space);
    close(a);
    let r = k.apply(kq, change(a as u64, EVFILT_READ, EV_DELETE, 0, 0, 0));
    k.require_failure(r, EBADF);
    close(b);
    close(kq);
}

#[test]
fn low_water_marks() {
    let mut k = Kq::new(0x7E56_0000_0000);
    let kq = k.kqueue();
    let (reader, writer) = k.socketpair(1);
    let rd = reader as u64;
    assert_eq!(k.apply(kq, change(rd, EVFILT_READ, EV_ADD, NOTE_LOWAT, 4, 0)), 0);
    send(writer, b"ab");
    assert!(k.poll(kq).is_empty());
    send(writer, b"cd");
    let ev = k.wait(kq);
    assert!(ev.len() == 1 && ev[0].ident == rd && ev[0].data == 4);
    assert_eq!(k.apply(kq, change(rd, EVFILT_READ, EV_ADD, NOTE_LOWAT, 8, 0)), 0);
    assert!(k.poll(kq).is_empty());
    assert_eq!(k.apply(kq, change(rd, EVFILT_READ, EV_ADD, 0, 0, 0)), 0);
    let ev = k.poll(kq);
    assert!(ev.len() == 1 && ev[0].data == 4);
    assert_eq!(k.apply(kq, change(rd, EVFILT_READ, EV_ADD, NOTE_LOWAT, 8, 0)), 0);
    assert!(k.poll(kq).is_empty());
    assert_eq!(k.apply(kq, change(writer as u64, EVFILT_WRITE, EV_ADD | EV_ONESHOT, NOTE_LOWAT, 1, 0)), 0);
    let ev = k.wait(kq);
    assert!(ev.len() == 1 && ev[0].filter == EVFILT_WRITE);
    // A write mark: fires while the free send space reaches it, silent
    // while the space is below it.
    assert_eq!(k.apply(kq, change(writer as u64, EVFILT_WRITE, EV_ADD | EV_ONESHOT, NOTE_LOWAT, 2, 0)), 0);
    let ev = k.wait(kq);
    assert!(ev.len() == 1 && ev[0].filter == EVFILT_WRITE && ev[0].data >= 2);
    assert_eq!(k.apply(kq, change(writer as u64, EVFILT_WRITE, EV_ADD | EV_ONESHOT, NOTE_LOWAT, 1 << 40, 0)), 0);
    assert!(k.poll(kq).iter().all(|e| e.filter != EVFILT_WRITE));
    assert_eq!(k.apply(kq, change(writer as u64, EVFILT_WRITE, EV_DELETE, 0, 0, 0)), 0);
    let r = k.apply(kq, change(rd, EVFILT_READ, EV_ADD, NOTE_LOWAT, 0, 0));
    k.require_failure(r, EINVAL);
    let (d0, d1) = k.socketpair(2);
    let r = k.apply(kq, change(d0 as u64, EVFILT_READ, EV_ADD, NOTE_LOWAT, 2, 0));
    k.require_failure(r, EINVAL);
    assert_eq!(k.apply(kq, change(d0 as u64, EVFILT_READ, EV_ADD, NOTE_LOWAT, 1, 0)), 0);
    send(d1, b"f");
    let ev = k.poll(kq);
    assert!(ev.iter().any(|e| e.ident == d0 as u64 && e.data == 1));
    close(d0);
    close(d1);
    close(writer);
    let ev = k.poll(kq);
    assert!(ev.len() == 1 && ev[0].ident == rd && ev[0].flags & EV_EOF != 0);
    close(reader);
    let (pr, pw) = fs::pipe(&k.ctx, k.base + 0x40);
    assert_eq!(k.apply(kq, change(pr as u64, EVFILT_READ, EV_ADD, NOTE_LOWAT, 64, 0)), 0);
    assert!(k.poll(kq).is_empty());
    k.write(pw, b"e");
    let ev = k.poll(kq);
    assert!(ev.len() == 1 && ev[0].data == 1);
    // The write end of a pipe: the free buffer space while the reader is
    // open, EV_EOF with no space once it is closed, and a new write
    // registration then fails with EPIPE.
    assert_eq!(k.apply(kq, change(pw as u64, EVFILT_WRITE, EV_ADD, 0, 0, 0)), 0);
    let ev = k.poll(kq);
    let w: Vec<_> = ev.iter().filter(|e| e.filter == EVFILT_WRITE).collect();
    let (free, _, _) = crate::filesystem::pipe::write_state(pw).unwrap();
    assert!(w.len() == 1 && w[0].ident == pw as u64 && w[0].data == free && free > 0 && w[0].flags & EV_EOF == 0);
    close(pr);
    let ev = k.poll(kq);
    assert!(ev.len() == 1 && ev[0].filter == EVFILT_WRITE && ev[0].flags & EV_EOF != 0 && ev[0].data == 0);
    assert_eq!(k.apply(kq, change(pw as u64, EVFILT_WRITE, EV_DELETE, 0, 0, 0)), 0);
    let r = k.apply(kq, change(pw as u64, EVFILT_WRITE, EV_ADD, 0, 0, 0));
    k.require_failure(r, 32);
    close(pw);
    close(kq);
}

/// Every registration a wait reports moves behind the waiting ones: level
/// events take turns one wait at a time, and an edge or dispatch event
/// triggered again comes after the one still waiting.
#[test]
fn reported_knotes_move_to_the_tail() {
    let mut k = Kq::new(0x7E5D_0000_0000);
    let kq = k.kqueue();
    let next = |k: &mut Kq| {
        let (r, ev) = k.kevent(kq, &[], 1, Some((0, 0)));
        assert_eq!(r, 1, "errno {}", k.errno());
        ev[0].ident
    };
    let (first_r, first_w) = fs::pipe(&k.ctx, k.base + 0x40);
    let (second_r, second_w) = fs::pipe(&k.ctx, k.base + 0x48);
    let (a, b) = k.socketpair(1);
    let level = [first_r as u64, second_r as u64, a as u64];
    for ident in level {
        assert_eq!(k.apply(kq, change(ident, EVFILT_READ, EV_ADD, 0, 0, 0)), 0);
    }
    k.write(first_w, b"a");
    k.write(second_w, b"b");
    send(b, b"c");
    for _ in 0..2 {
        for ident in level {
            assert_eq!(next(&mut k), ident);
        }
    }
    for ident in level {
        assert_eq!(k.apply(kq, change(ident, EVFILT_READ, EV_DELETE, 0, 0, 0)), 0);
    }

    for ident in [11, 12] {
        assert_eq!(k.apply(kq, change(ident, EVFILT_USER, EV_ADD | EV_CLEAR, 0, 0, 0)), 0);
        assert_eq!(k.apply(kq, change(ident, EVFILT_USER, 0, NOTE_TRIGGER, 0, 0)), 0);
    }
    assert_eq!(next(&mut k), 11);
    assert_eq!(k.apply(kq, change(11, EVFILT_USER, 0, NOTE_TRIGGER, 0, 0)), 0);
    assert_eq!(next(&mut k), 12);
    assert_eq!(next(&mut k), 11);
    assert!(k.poll(kq).is_empty());

    for ident in [21, 22] {
        assert_eq!(k.apply(kq, change(ident, EVFILT_USER, EV_ADD | EV_DISPATCH, 0, 0, 0)), 0);
        assert_eq!(k.apply(kq, change(ident, EVFILT_USER, 0, NOTE_TRIGGER, 0, 0)), 0);
    }
    assert_eq!(next(&mut k), 21);
    assert_eq!(k.apply(kq, change(21, EVFILT_USER, EV_ENABLE, NOTE_TRIGGER, 0, 0)), 0);
    assert_eq!(next(&mut k), 22);
    assert_eq!(next(&mut k), 21);
    assert!(k.poll(kq).is_empty());

    for fd in [first_r, first_w, second_r, second_w, a, b, kq] {
        close(fd);
    }
}

#[test]
fn disable_and_dispatch() {
    let mut k = Kq::new(0x7E57_0000_0000);
    let kq = k.kqueue();
    let (a, b) = k.socketpair(1);
    let id = a as u64;
    let marker = 0x42;
    assert_eq!(k.apply(kq, change(id, EVFILT_READ, EV_ADD | EV_DISABLE, 0, 0, 0)), 0);
    send(b, b"a");
    assert!(k.poll(kq).is_empty());
    assert_eq!(k.apply(kq, change(id, EVFILT_READ, EV_ADD | EV_ENABLE, 0, 0, 0)), 0);
    let ev = k.poll(kq);
    assert!(ev.len() == 1 && ev[0].ident == id && ev[0].data == 1);
    assert_eq!(k.apply(kq, change(id, EVFILT_READ, EV_ADD | EV_DISABLE, 0, 0, 0)), 0);
    assert!(k.poll(kq).is_empty());
    assert_eq!(k.apply(kq, change(id, EVFILT_READ, EV_DELETE, 0, 0, 0)), 0);
    assert_eq!(recv(a, 1), 1);
    assert_eq!(k.apply(kq, change(id, EVFILT_READ, EV_ADD | EV_DISPATCH, 0, 0, marker)), 0);
    send(b, b"bc");
    let ev = k.wait(kq);
    assert!(ev.len() == 1 && ev[0].udata == marker && ev[0].data == 2);
    assert!(k.poll(kq).is_empty());
    send(b, b"d");
    assert!(k.poll(kq).is_empty());
    assert_eq!(k.apply(kq, change(id, EVFILT_READ, EV_ADD | EV_ENABLE, 0, 0, 0)), 0);
    let ev = k.wait(kq);
    assert!(ev.len() == 1 && ev[0].data == 3);
    assert!(k.poll(kq).is_empty());
    assert_eq!(k.apply(kq, change(id, EVFILT_READ, EV_DELETE, 0, 0, 0)), 0);
    assert_eq!(k.apply(kq, change(id, EVFILT_READ, EV_ADD | EV_FORCEONESHOT, 0, 0, 0)), 0);
    assert_eq!(k.poll(kq)[0].data, 3);
    assert_eq!(k.poll(kq)[0].data, 3);
    assert_eq!(k.apply(kq, change(id, EVFILT_READ, EV_ADD | EV_FORCEONESHOT, 0, 0, 0)), 0);
    assert_eq!(k.poll(kq)[0].data, 3);
    assert!(k.poll(kq).is_empty());
    let r = k.apply(kq, change(id, EVFILT_READ, EV_DELETE, 0, 0, 0));
    k.require_failure(r, ENOENT);
    assert_eq!(recv(a, 4), 3);
    assert_eq!(k.apply(kq, change(9, EVFILT_USER, EV_ADD | EV_DISABLE, 0, 0, 0)), 0);
    assert_eq!(k.apply(kq, change(9, EVFILT_USER, 0, NOTE_TRIGGER, 0, 0)), 0);
    assert!(k.poll(kq).is_empty());
    assert_eq!(k.apply(kq, change(9, EVFILT_USER, EV_ADD | EV_ENABLE, 0, 0, 0)), 0);
    let ev = k.poll(kq);
    assert!(ev.len() == 1 && ev[0].filter == EVFILT_USER);
    assert_eq!(k.apply(kq, change(9, EVFILT_USER, EV_DELETE, 0, 0, 0)), 0);
    close(a);
    close(b);
    close(kq);
}

#[test]
fn errors() {
    let mut k = Kq::new(0x7E58_0000_0000);
    let kq = k.kqueue();
    let unknown = 0x1000_0000 + 4000;
    let changes = [
        change(unknown, EVFILT_READ, EV_ADD, 0, 0, 0),
        change(8, EVFILT_USER, EV_DELETE, 0, 0, 0),
        change(9, EVFILT_USER, EV_ADD | EV_ENABLE | EV_DISABLE, 0, 0, 0),
    ];
    let (n, ev) = k.kevent(kq, &changes, 4, None);
    assert_eq!(n, 3);
    assert!(ev[0].flags == EV_ERROR && ev[0].data == EBADF as u64 && ev[0].ident == unknown);
    assert!(ev[1].flags == EV_ERROR && ev[1].data == ENOENT as u64);
    assert!(ev[2].flags == EV_ERROR && ev[2].data == EINVAL as u64);
    let (r, ev) = k.kevent(kq, &changes, 1, None);
    k.require_failure(r, ENOENT);
    let first = event_from_bytes(&k.ctx.read_bytes(k.base + EVENTS, 0x20).unwrap());
    assert!(ev.is_empty() && first.flags == EV_ERROR && first.data == EBADF as u64);
    let r = k.kevent(kq, &changes, 0, None).0;
    k.require_failure(r, EBADF);
    let r = k.apply(kq, change(kq as u64, EVFILT_READ, EV_ADD, 0, 0, 0));
    k.require_failure(r, EINVAL);
    let r = k.apply(kq, change(1, 1, EV_ADD, 0, 0, 0));
    k.require_failure(r, EINVAL);
    let r = k.apply(kq, change(1, EVFILT_PROC, EV_ADD, 0, 0, 0));
    k.require_failure(r, EINVAL);
    assert_eq!(k.apply(kq, change(1, EVFILT_USER, EV_ADD | EV_DISABLE, 0, 0, 0)), 0);
    assert_eq!(k.apply(kq, change(1, EVFILT_USER, EV_DELETE, 0, 0, 0)), 0);
    let r = k.call(cb_kevent, &[kq as u64, 0, 0, k.base + EVENTS, u32::MAX as u64, 0]);
    k.require_failure(r, EINVAL);
    let (pr, pw) = fs::pipe(&k.ctx, k.base + 0x40);
    for c in [
        change(pw as u64, EVFILT_READ, EV_ADD, 0, 0, 0),
        change(pr as u64, EVFILT_WRITE, EV_ADD, 0, 0, 0),
        change(pr as u64, EVFILT_READ, EV_ADD | EV_CLEAR, 0, 0, 0),
    ] {
        let r = k.apply(kq, c);
        k.require_failure(r, EINVAL);
    }
    close(pr);
    close(pw);
    assert_eq!(k.kevent(kq, &[change(1, 0, EV_ADD, 0, 0, 0)], 4, Some((0, 0))).0, 0);
    let r = k.kevent(kq, &[], 4, Some((0, 1_000_000_000))).0;
    k.require_failure(r, EINVAL);
    let r = k.kevent(kq, &[], 4, Some((-1, 0))).0;
    k.require_failure(r, EINVAL);
    close(kq);
}

#[test]
fn file_watch() {
    let mut k = Kq::new(0x7E59_0000_0000);
    let path = temp_path("file");
    let file = fs::open_host_file(&path);
    let kq = k.kqueue();
    let marker = 0x31;
    let id = file as u64;
    let notes = NOTE_DELETE | NOTE_WRITE | NOTE_EXTEND | NOTE_LINK | NOTE_RENAME;
    assert_eq!(k.apply(kq, change(id, EVFILT_VNODE, 0x25, notes, 0, marker)), 0);
    assert!(k.poll(kq).is_empty());
    k.write(file, b"abc");
    let ev = k.wait(kq);
    assert!(ev.len() == 1 && ev[0].ident == id && ev[0].filter == EVFILT_VNODE && ev[0].flags == EV_CLEAR && ev[0].udata == marker);
    assert!(ev[0].fflags == NOTE_WRITE | NOTE_EXTEND && ev[0].data == 0);
    assert!(k.poll(kq).is_empty());
    std::fs::OpenOptions::new().write(true).open(&path).unwrap().set_len(0).unwrap();
    settle();
    assert!(k.poll(kq).is_empty());
    assert_eq!(k.apply(kq, change(id, EVFILT_VNODE, EV_ADD, NOTE_ATTRIB, 0, 0)), 0);
    k.write(file, b"abc");
    settle();
    std::fs::OpenOptions::new().write(true).open(&path).unwrap().set_len(0).unwrap();
    let ev = k.wait(kq);
    assert!(ev.len() == 1 && ev[0].fflags == NOTE_ATTRIB && ev[0].udata == 0);
    let r = k.apply(kq, change(id, EVFILT_VNODE, EV_ADD, 0x40, 0, 0));
    k.require_failure(r, EINVAL);
    assert_eq!(k.apply(kq, change(id, EVFILT_VNODE, EV_DELETE, 0, 0, 0)), 0);
    let r = k.apply(kq, change(id, EVFILT_VNODE, EV_DELETE, 0, 0, 0));
    k.require_failure(r, ENOENT);

    // sceKernelAddFileEvent / sceKernelDeleteFileEvent on an event queue.
    let eq = equeue::create();
    assert_eq!(k.call(cb_add_file_event, &[eq, id, NOTE_WRITE as u64, marker]), OK);
    k.write(file, b"defg");
    let ev = equeue::wait(eq, 4, Some(1_000_000)).unwrap();
    assert!(ev.len() == 1 && ev[0].ident == id && ev[0].filter == EVFILT_VNODE);
    assert!(ev[0].fflags == NOTE_WRITE | NOTE_EXTEND && ev[0].udata == marker);
    assert_eq!(k.call(cb_delete_file_event, &[eq, id]), OK);
    assert_eq!(k.call(cb_delete_file_event, &[eq, id]), ERR_NOT_FOUND);
    assert_eq!(k.call(cb_add_file_event, &[eq, 0x1000_0FA0, NOTE_WRITE as u64, 0]), 0x8002_0009u32 as i32 as i64);
    let (a, b) = k.socketpair(1);
    assert_eq!(k.call(cb_add_file_event, &[eq, a as u64, NOTE_WRITE as u64, 0]), 0x8002_0016u32 as i32 as i64);
    let r = k.apply(kq, change(a as u64, EVFILT_VNODE, EV_ADD, NOTE_WRITE, 0, 0));
    k.require_failure(r, EINVAL);
    let r = k.apply(kq, change(kq as u64, EVFILT_VNODE, EV_ADD, NOTE_WRITE, 0, 0));
    k.require_failure(r, EINVAL);
    close(a);
    close(b);
    let (pr, pw) = fs::pipe(&k.ctx, k.base + 0x40);
    let r = k.apply(kq, change(pr as u64, EVFILT_VNODE, EV_ADD, NOTE_WRITE, 0, 0));
    k.require_failure(r, EINVAL);
    close(pr);
    close(pw);
    assert_eq!(k.apply(kq, change(id, EVFILT_VNODE, EV_ADD, NOTE_WRITE, 0, 0)), 0);
    assert_eq!(k.call(cb_add_file_event, &[eq, id, NOTE_WRITE as u64, 0]), OK);
    close(file);
    let r = k.apply(kq, change(id, EVFILT_VNODE, EV_DELETE, 0, 0, 0));
    k.require_failure(r, EBADF);
    assert_eq!(k.call(cb_delete_file_event, &[eq, id]), 0x8002_0009u32 as i32 as i64);
    assert!(equeue::delete(eq));
    close(kq);
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn rename_and_delete_watched_file() {
    let mut k = Kq::new(0x7E5A_0000_0000);
    let path = temp_path("moved");
    let renamed = path.with_extension("renamed");
    let file = fs::open_host_file(&path);
    let kq = k.kqueue();
    let id = file as u64;
    assert_eq!(k.apply(kq, change(id, EVFILT_VNODE, EV_ADD | EV_CLEAR, NOTE_DELETE | NOTE_RENAME, 0, 0)), 0);
    std::fs::rename(&path, &renamed).unwrap();
    settle();
    let ev = k.poll(kq);
    assert!(ev.len() == 1 && ev[0].ident == id && ev[0].fflags == NOTE_RENAME);
    std::fs::remove_file(&renamed).unwrap();
    settle();
    let ev = k.poll(kq);
    assert!(ev.len() == 1 && ev[0].ident == id && ev[0].fflags == NOTE_DELETE);
    close(file);
    close(kq);
}

#[test]
fn directory_watch() {
    let mut k = Kq::new(0x7E5B_0000_0000);
    let path = temp_path("directory");
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir(&path).unwrap();
    let dir = fs::open_host_dir(&path);
    let kq = k.kqueue();
    let id = dir as u64;
    assert_eq!(k.apply(kq, change(id, EVFILT_VNODE, EV_ADD | EV_CLEAR, NOTE_WRITE | NOTE_LINK | NOTE_DELETE | NOTE_RENAME, 0, 0)), 0);
    assert!(k.poll(kq).is_empty());
    std::fs::write(path.join("child.bin"), b"").unwrap();
    let ev = k.wait(kq);
    assert!(ev.len() == 1 && ev[0].ident == id && ev[0].fflags == NOTE_WRITE);
    std::fs::create_dir(path.join("sub")).unwrap();
    let ev = k.wait(kq);
    assert!(ev.len() == 1 && ev[0].fflags == NOTE_WRITE | NOTE_LINK);
    std::fs::rename(path.join("child.bin"), path.join("renamed.bin")).unwrap();
    settle();
    let ev = k.poll(kq);
    assert!(ev.len() == 1 && ev[0].fflags == NOTE_WRITE);
    std::fs::remove_file(path.join("renamed.bin")).unwrap();
    settle();
    let ev = k.poll(kq);
    assert!(ev.len() == 1 && ev[0].fflags == NOTE_WRITE);
    std::fs::remove_dir(path.join("sub")).unwrap();
    settle();
    let ev = k.poll(kq);
    assert!(ev.len() == 1 && ev[0].fflags == NOTE_WRITE | NOTE_LINK);
    close(dir);
    close(kq);
    std::fs::remove_dir_all(&path).unwrap();
}

#[test]
fn delete_watched_file() {
    let mut k = Kq::new(0x7E5C_0000_0000);
    let path = temp_path("deleted");
    let file = fs::open_host_file(&path);
    let kq = k.kqueue();
    let id = file as u64;
    assert_eq!(k.apply(kq, change(id, EVFILT_VNODE, EV_ADD | EV_CLEAR, NOTE_DELETE | NOTE_WRITE | NOTE_RENAME, 0, 0)), 0);
    assert!(k.poll(kq).is_empty());
    std::fs::remove_file(&path).unwrap();
    let ev = k.wait(kq);
    assert!(ev.len() == 1 && ev[0].ident == id && ev[0].fflags == NOTE_DELETE);
    settle();
    assert!(k.poll(kq).is_empty());
    close(file);
    close(kq);
}

#[test]
fn nids_match_names() {
    for &(nid, name, _, _) in EXPORTS {
        assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
    }
}
