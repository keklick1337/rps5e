// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Guest signal delivery tests: real guest code on real guest threads.

#[cfg(windows)]
#[allow(unused_imports)]
use crate::host::c as libc;
use super::*;
use ps5cpu::exec::{GuestExit, HleDispatcher, ThreadStart};
use std::sync::mpsc;

const CODE: u64 = 0x7C10_0000_0000;
const DATA: u64 = 0x7C10_0001_0000;
const STACKS: u64 = 0x7C10_1000_0000;
const STACK_SIZE: u64 = 0x10_0000;

const SPIN: u64 = CODE;
const HANDLER: u64 = CODE + 0x100;
/// Calls `WAIT_STUB` and returns r12.
const CALL_WAIT: u64 = CODE + 0x200;
/// Calls `BUSY_STUB` and returns r12.
const CALL_BUSY: u64 = CODE + 0x240;
/// Calls `SELF_STUB` and returns its result.
const CALL_SELF: u64 = CODE + 0x280;
const WAIT_STUB: u64 = CODE + 0x300;
const BUSY_STUB: u64 = CODE + 0x310;
const SELF_STUB: u64 = CODE + 0x320;

fn caller(stub: u64, return_r12: bool) -> Vec<u8> {
    let mut v = vec![0x48, 0xB8]; // mov rax, imm64
    v.extend_from_slice(&stub.to_le_bytes());
    v.extend_from_slice(&[0xFF, 0xD0]); // call rax
    if return_r12 {
        v.extend_from_slice(&[0x4C, 0x89, 0xE0]); // mov rax, r12
    }
    v.push(0xC3);
    v
}

fn map_rwx(addr: u64, size: u64) {
    let p = unsafe {
        libc::mmap(
            addr as *mut libc::c_void,
            size as usize,
            libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_FIXED_NOREPLACE,
            -1,
            0,
        )
    };
    assert_eq!(p as u64, addr, "map {addr:#x}");
    ps5vm::ranges::track(addr, size, ps5vm::MemProtect::RWX);
}

fn write(addr: u64, bytes: &[u8]) {
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), addr as *mut u8, bytes.len()) };
}

fn rd(addr: u64) -> u64 {
    unsafe { std::ptr::read_volatile(addr as *const u64) }
}

static WAIT_GATE: Mutex<()> = Mutex::new(());
static WAIT_CV: Condvar = Condvar::new();

struct TestDispatcher;

impl HleDispatcher for TestDispatcher {
    fn dispatch_at(&self, stub: u64, regs: &[u64; 16], _rsp: u64, _rip: u64) -> Option<u64> {
        let flag = regs[7]; // rdi
        match stub {
            // Block in an interruptible HLE wait until the handler set the flag.
            WAIT_STUB => {
                let deadline = Instant::now() + Duration::from_secs(5);
                let mut g = WAIT_GATE.lock().unwrap();
                while rd(flag) == 0 {
                    let (ng, timed_out) = wait_cv(&WAIT_GATE, &WAIT_CV, g, Some(deadline));
                    g = ng;
                    if timed_out {
                        break;
                    }
                }
                Some(rd(flag))
            }
            // Busy in host code (no safe point) for a while.
            BUSY_STUB => {
                std::thread::sleep(Duration::from_millis(100));
                Some(0)
            }
            SELF_STUB => {
                let r = raise_exception(current(), SIGUSR1);
                assert_eq!(r, OK);
                Some(rd(flag))
            }
            _ => None,
        }
    }
}

fn setup() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        ps5cpu::exec::init();
        ps5cpu::exec::set_dispatcher(Box::new(TestDispatcher));
        map_rwx(CODE, 0x1000);
        map_rwx(DATA, 0x1000);
        // spin(flag*): while (*flag == 0); return r12;
        write(SPIN, &[0x48, 0x8B, 0x07, 0x48, 0x85, 0xC0, 0x74, 0xF8, 0x4C, 0x89, 0xE0, 0xC3]);
        // handler(signum, uc):
        //   flag = uc->mc_rdi; flag[0] = signum; flag[1] = uc->mc_rsp;
        //   uc->mc_r12 = 0x77
        write(
            HANDLER,
            &[
                0x48, 0x8B, 0x46, 0x48, // mov rax, [rsi+0x48]
                0x48, 0x8B, 0x8E, 0xF8, 0x00, 0x00, 0x00, // mov rcx, [rsi+0xF8]
                0x48, 0x89, 0x48, 0x08, // mov [rax+8], rcx
                0x48, 0x89, 0x38, // mov [rax], rdi
                0x48, 0xC7, 0x86, 0xA0, 0x00, 0x00, 0x00, 0x77, 0x00, 0x00, 0x00, // mov qword [rsi+0xA0], 0x77
                0xC3,
            ],
        );
        write(CALL_WAIT, &caller(WAIT_STUB, true));
        write(CALL_BUSY, &caller(BUSY_STUB, true));
        write(CALL_SELF, &caller(SELF_STUB, false));
        write(WAIT_STUB, &[0xCC]);
        write(BUSY_STUB, &[0xCC]);
        write(SELF_STUB, &[0xCC]);
        ps5cpu::exec::add_guest_code_range(CODE, 0x1000, "signal-test");
        for i in 0..5 {
            ps5cpu::exec::map_guest_stack(STACKS + (i + 1) * 0x1000_0000, STACK_SIZE).unwrap();
        }
        let vm: &'static ps5vm::VirtualMemory = Box::leak(Box::new(ps5vm::VirtualMemory::new()));
        let mut ctx = CpuContext::new(vm);
        ctx.set(crate::context::CpuRegister::Rdi, SIGUSR1 as u64);
        ctx.set(crate::context::CpuRegister::Rsi, HANDLER);
        assert_eq!(cb_install_exception_handler(&mut ctx), OK);
        assert_eq!(cb_install_exception_handler(&mut ctx), EAGAIN);
        ctx.set(crate::context::CpuRegister::Rdi, 2);
        assert_eq!(cb_install_exception_handler(&mut ctx), EINVAL);
    });
}

/// Start guest thread `slot` running `entry(flag)`; returns a receiver for
/// its exit value once it has registered.
fn start(slot: u64, handle: u64, entry: u64, flag: u64) -> mpsc::Receiver<u64> {
    let (tx, rx) = mpsc::channel();
    let (ready_tx, ready_rx) = mpsc::channel();
    let top = STACKS + (slot + 1) * 0x1000_0000;
    ps5cpu::exec::spawn_guest_thread(
        ThreadStart {
            name: format!("sig-test-{slot}"),
            entry,
            arg: flag,
            stack_top: top,
            tp: 0,
            on_start: Some(Box::new(move || {
                register_thread(handle);
                ready_tx.send(()).unwrap();
            })),
        },
        Box::new(move |exit| {
            unregister_thread();
            let v = match exit {
                GuestExit::Returned(v) | GuestExit::ThreadExit(v) => v,
            };
            tx.send(v).unwrap();
        }),
    )
    .unwrap();
    ready_rx.recv().unwrap();
    rx
}

fn in_stack(slot: u64, addr: u64) -> bool {
    let top = STACKS + (slot + 1) * 0x1000_0000;
    (top - STACK_SIZE..top).contains(&addr)
}

#[test]
fn ucontext_layout() {
    use ps5cpu::signal::OrbisUcontext;
    use std::mem::offset_of;
    let mc = offset_of!(OrbisUcontext, uc_mcontext);
    assert_eq!(std::mem::size_of::<OrbisUcontext>(), 0x500);
    assert_eq!(mc, 0x40);
    assert_eq!(mc + offset_of!(ps5cpu::signal::OrbisMcontext, mc_rdi), 0x48);
    assert_eq!(mc + offset_of!(ps5cpu::signal::OrbisMcontext, mc_rax), 0x78);
    assert_eq!(mc + offset_of!(ps5cpu::signal::OrbisMcontext, mc_r12), 0xA0);
    assert_eq!(mc + offset_of!(ps5cpu::signal::OrbisMcontext, mc_rip), 0xE0);
    assert_eq!(mc + offset_of!(ps5cpu::signal::OrbisMcontext, mc_rflags), 0xF0);
    assert_eq!(mc + offset_of!(ps5cpu::signal::OrbisMcontext, mc_rsp), 0xF8);
    assert_eq!(mc + offset_of!(ps5cpu::signal::OrbisMcontext, mc_len), 0x108);
    assert_eq!(mc + offset_of!(ps5cpu::signal::OrbisMcontext, mc_fpstate), 0x140);
    assert_eq!(mc + offset_of!(ps5cpu::signal::OrbisMcontext, mc_fsbase), 0x480);
    assert_eq!(offset_of!(OrbisUcontext, uc_link), 0x4C0);
}

#[test]
fn raise_argument_checks() {
    assert_eq!(raise_exception(0, SIGUSR1), EINVAL);
    assert_eq!(raise_exception(0x1234, SIGSEGV), EINVAL);
}

/// One test drives every delivery path in sequence (the handler table and
/// dispatcher are process-wide).
#[test]
fn delivery_paths() {
    let _mode = WAIT_MODE.lock().unwrap_or_else(|e| e.into_inner());
    setup();

    // 1. Asynchronous: the target spins in guest code. The handler runs on
    //    the interrupted frame and its context change (r12) is applied.
    let flag = DATA;
    let rx = start(0, 0xA000, SPIN, flag);
    std::thread::sleep(Duration::from_millis(20));
    assert_eq!(raise_exception(0xA000, SIGUSR1), OK);
    let v = rx.recv_timeout(Duration::from_secs(5)).expect("spinning thread was not interrupted");
    assert_eq!(rd(flag), SIGUSR1 as u64);
    assert!(in_stack(0, rd(flag + 8)), "mc_rsp {:#x} not on the guest stack", rd(flag + 8));
    assert_eq!(v, 0x77, "handler context change not applied");

    // 2. Inside an HLE wait: delivered within the wait, quickly, with the
    //    import's guest context (read-only: r12 stays 0).
    let flag = DATA + 0x40;
    let rx = start(1, 0xB000, CALL_WAIT, flag);
    std::thread::sleep(Duration::from_millis(30));
    let t0 = Instant::now();
    assert_eq!(raise_exception(0xB000, SIGUSR1), OK);
    let v = rx.recv_timeout(Duration::from_secs(5)).expect("waiting thread did not return");
    let elapsed = t0.elapsed();
    assert_eq!(rd(flag), SIGUSR1 as u64);
    assert!(in_stack(1, rd(flag + 8)));
    assert_eq!(v, 0, "an in-wait context must not be applied");
    assert!(elapsed < Duration::from_millis(100), "in-wait delivery took {elapsed:?}");

    // 2b. The same with the condvar published under the table lock.
    set_fast_wait(false);
    let flag = DATA + 0x100;
    let rx = start(4, 0xB100, CALL_WAIT, flag);
    std::thread::sleep(Duration::from_millis(30));
    let t0 = Instant::now();
    assert_eq!(raise_exception(0xB100, SIGUSR1), OK);
    let v = rx.recv_timeout(Duration::from_secs(5)).expect("waiting thread did not return");
    let elapsed = t0.elapsed();
    set_fast_wait(true);
    assert_eq!(rd(flag), SIGUSR1 as u64);
    assert_eq!(v, 0);
    assert!(elapsed < Duration::from_millis(100), "in-wait delivery (table lock) took {elapsed:?}");

    // 3. Busy in host code without a safe point: taken when the import
    //    returns, with the context applied.
    let flag = DATA + 0x80;
    let rx = start(2, 0xC000, CALL_BUSY, flag);
    std::thread::sleep(Duration::from_millis(20));
    assert_eq!(raise_exception(0xC000, SIGUSR1), OK);
    let v = rx.recv_timeout(Duration::from_secs(5)).expect("busy thread did not return");
    assert_eq!(rd(flag), SIGUSR1 as u64);
    assert_eq!(v, 0x77, "import-return context change not applied");

    // 4. Raise on the calling thread: synchronous.
    let flag = DATA + 0xC0;
    let rx = start(3, 0xD000, CALL_SELF, flag);
    let v = rx.recv_timeout(Duration::from_secs(5)).expect("self-raising thread did not return");
    assert_eq!(v, SIGUSR1 as u64);

    // 5. Unknown target thread.
    assert_eq!(raise_exception(0xE000, SIGUSR1), ESRCH);
}

#[test]
fn export_nids() {
    use crate::sync::test_util::nid;
    assert_eq!(nid("sceKernelInstallExceptionHandler"), "WkwEd3N7w0Y");
    assert_eq!(nid("sceKernelRemoveExceptionHandler"), "Qhv5ARAoOEc");
    assert_eq!(nid("sceKernelRaiseException"), "il03nluKfMk");
}

#[test]
fn interruptible_sleep_outside_guest_threads() {
    let t0 = Instant::now();
    sleep(Duration::from_millis(15));
    assert!(t0.elapsed() >= Duration::from_millis(15));
}

/// Serializes the tests that switch how waits publish their condvar.
static WAIT_MODE: Mutex<()> = Mutex::new(());

/// `wake_thread` wakes a registered thread out of an interruptible sleep at
/// once, whether the wait is published lock-free or under the table lock.
#[test]
fn wake_thread_ends_a_published_wait() {
    let _g = WAIT_MODE.lock().unwrap_or_else(|e| e.into_inner());
    for (i, fast) in [true, false].into_iter().enumerate() {
        set_fast_wait(fast);
        let handle = 0xF100 + i as u64 * 0x100;
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (tx, rx) = mpsc::channel();
        let flag = stop.clone();
        let t = std::thread::spawn(move || {
            register_thread(handle);
            TEST_SLICE.with(|s| s.set(Some(Duration::from_secs(60))));
            tx.send(()).unwrap();
            let t0 = Instant::now();
            sleep_until_unless(Instant::now() + Duration::from_secs(30), || flag.load(Ordering::SeqCst));
            unregister_thread();
            t0.elapsed()
        });
        rx.recv().unwrap();
        std::thread::sleep(Duration::from_millis(30));
        stop.store(true, Ordering::SeqCst);
        wake_thread(handle);
        let slept = t.join().unwrap();
        assert!(slept < Duration::from_secs(5), "fast={fast}: the sleeper was not woken ({slept:?})");
    }
    set_fast_wait(true);
}

/// A thread waits many times on short-lived condvars while another wakes it
/// in a loop: the waker never notifies a condvar after the waiter withdrew
/// it (it would touch freed memory), and every wait ends.
#[test]
fn waker_and_short_lived_condvars() {
    let _g = WAIT_MODE.lock().unwrap_or_else(|e| e.into_inner());
    for (i, fast) in [true, false].into_iter().enumerate() {
        set_fast_wait(fast);
        let handle = 0xF400 + i as u64 * 0x100;
        let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (tx, rx) = mpsc::channel();
        let flag = done.clone();
        let t = std::thread::spawn(move || {
            register_thread(handle);
            tx.send(()).unwrap();
            for _ in 0..20_000 {
                let pair = Box::new((Mutex::new(0u64), Condvar::new()));
                let g = pair.0.lock().unwrap();
                let (mut g, _) = wait_cv(&pair.0, &pair.1, g, Some(Instant::now() + Duration::from_micros(20)));
                *g += 1;
                drop(g);
                drop(pair);
            }
            flag.store(true, Ordering::SeqCst);
            unregister_thread();
        });
        rx.recv().unwrap();
        let mut wakes = 0u64;
        while !done.load(Ordering::SeqCst) {
            wake_thread(handle);
            wakes += 1;
        }
        t.join().unwrap();
        assert!(wakes > 0);
    }
    set_fast_wait(true);
}
