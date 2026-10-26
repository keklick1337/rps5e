// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! End-to-end fiber tests: guest-style code runs natively on mapped guest
//! stacks and calls libSceFiber through real INT3 import stubs, so every
//! switch goes through the trap handler, the HLE callback and the guest-side
//! switch routine exactly as in a game.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use ps5hle::context::CpuContext;
use ps5hle::registry::HleCallback;

const MEM: u64 = 0x5B00_0000_0000;
const MEM_SIZE: u64 = 0x40_0000;
const STACKS: u64 = 0x5B10_0000_0000;

fn vm() -> &'static ps5vm::VirtualMemory {
    static VM: OnceLock<ps5vm::VirtualMemory> = OnceLock::new();
    VM.get_or_init(|| {
        let vm = ps5vm::VirtualMemory::new();
        vm.map(MEM, MEM_SIZE, ps5vm::MemProtect::RW).unwrap();
        vm
    })
}

const NAMES: [&str; 6] = [
    "_sceFiberInitializeImpl",
    "sceFiberRun",
    "sceFiberSwitch",
    "sceFiberReturnToThread",
    "sceFiberGetSelf",
    "sceFiberFinalize",
];

struct Stubs {
    base: u64,
    cbs: Vec<HleCallback>,
}

struct Dispatcher;

impl ps5cpu::exec::HleDispatcher for Dispatcher {
    fn dispatch_at(&self, stub: u64, regs: &[u64; 16], _rsp: u64, _rip: u64) -> Option<u64> {
        let s = stubs();
        let idx = ((stub - s.base) / 16) as usize;
        let cb = *s.cbs.get(idx)?;
        if std::env::var_os("FIBER_TEST_TRACE").is_some() {
            eprintln!("[trap] {} rdi={:#x} rsi={:#x} rsp={:#x}", NAMES[idx], regs[7], regs[6], regs[4]);
        }
        let mut ctx = CpuContext::new(vm());
        ctx.regs.copy_from_slice(regs);
        Some(cb(&mut ctx) as u64)
    }
}

fn stubs() -> &'static Stubs {
    static S: OnceLock<Stubs> = OnceLock::new();
    S.get_or_init(|| {
        ps5cpu::exec::init();
        vm();
        let base = ps5vm::os::map_anon(0x1000, ps5vm::MemProtect::RWX).expect("map the stub page");
        unsafe { std::ptr::write_bytes(base as *mut u8, 0xCC, 0x1000) };
        let mut reg = ps5hle::registry::HleRegistry::new();
        ps5hle::fiber::register(&mut reg);
        let cbs = NAMES.iter().map(|n| ps5hle::fiber::export(n).unwrap()).collect();
        ps5cpu::exec::set_dispatcher(Box::new(Dispatcher));
        Stubs { base, cbs }
    })
}

fn stub(name: &str) -> u64 {
    stubs().base + 16 * NAMES.iter().position(|n| *n == name).unwrap() as u64
}

type F3 = extern "sysv64" fn(u64, u64, u64) -> i32;
type F2 = extern "sysv64" fn(u64, u64) -> i32;

fn f3(name: &str) -> F3 {
    unsafe { std::mem::transmute::<u64, F3>(stub(name)) }
}
fn f2(name: &str) -> F2 {
    unsafe { std::mem::transmute::<u64, F2>(stub(name)) }
}

/// Guest-memory bump allocator.
fn galloc(size: u64) -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(MEM);
    let a = NEXT.fetch_add((size + 0xF) & !0xF, Ordering::SeqCst);
    assert!(a + size <= MEM + MEM_SIZE);
    a
}

fn init_fiber(name: &str, entry: extern "sysv64" fn(u64, u64), arg: u64, stack: u64) -> u64 {
    let fiber = galloc(0x100);
    let name_ptr = galloc(32);
    unsafe {
        std::ptr::copy_nonoverlapping(name.as_ptr(), name_ptr as *mut u8, name.len());
        *((name_ptr + name.len() as u64) as *mut u8) = 0;
    }
    let ctx_addr = if stack != 0 { galloc(stack) } else { 0 };
    // Initialize has 8 arguments: the last two go on the stack.
    type Init = extern "sysv64" fn(u64, u64, u64, u64, u64, u64, u64, u32) -> i32;
    let init: Init = unsafe { std::mem::transmute(stub("_sceFiberInitializeImpl")) };
    assert_eq!(init(fiber, name_ptr, entry as usize as u64, arg, ctx_addr, stack, 0, 0x0400_0000), 0);
    fiber
}

/// Import stubs may only be called from guest threads (a trap outside guest
/// code is a host fault).
/// Run `f` as a guest thread (own host thread, mapped guest stack).
fn guest_thread(slot: u64, f: extern "sysv64" fn(u64) -> u64, arg: u64) -> u64 {
    let top = STACKS + (slot + 1) * 0x10_0000;
    ps5cpu::exec::map_guest_stack(top, 0x8_0000).unwrap();
    let out = std::sync::Arc::new(Mutex::new(None));
    let out2 = out.clone();
    let start = ps5cpu::exec::ThreadStart {
        name: format!("fiber-test-{slot}"),
        entry: f as usize as u64,
        arg,
        stack_top: top,
        tp: 0,
        on_start: None,
    };
    let h = ps5cpu::exec::spawn_guest_thread(
        start,
        Box::new(move |e| {
            *out2.lock().unwrap() = Some(e);
        }),
    )
    .unwrap();
    h.join().unwrap();
    ps5cpu::exec::unmap_guest_stack(top, 0x8_0000);
    let exit = out.lock().unwrap().take().unwrap();
    match exit {
        ps5cpu::exec::GuestExit::Returned(v) => v,
        e => panic!("guest thread ended with {e:?}"),
    }
}

// ---------------------------------------------------------------------------
// Round trip: thread -> A -> B -> thread -> B -> A -> thread
// ---------------------------------------------------------------------------

static LOG: Mutex<Vec<(char, u64)>> = Mutex::new(Vec::new());
static FIBER_A: AtomicU64 = AtomicU64::new(0);
static FIBER_B: AtomicU64 = AtomicU64::new(0);

fn log(c: char, v: u64) {
    LOG.lock().unwrap().push((c, v));
}

extern "sysv64" fn entry_a(init: u64, run: u64) {
    log('A', init);
    log('a', run);
    let out = galloc(8);
    let mut me = galloc(8);
    assert_eq!(f2("sceFiberGetSelf")(me, 0), 0);
    me = unsafe { *(me as *const u64) };
    log('s', (me == FIBER_A.load(Ordering::SeqCst)) as u64);
    assert_eq!(f3("sceFiberSwitch")(FIBER_B.load(Ordering::SeqCst), 2, out), 0);
    log('x', unsafe { *(out as *const u64) });
    assert_eq!(f2("sceFiberReturnToThread")(6, out), 0);
    log('z', unsafe { *(out as *const u64) });
    loop {
        f2("sceFiberReturnToThread")(99, 0);
    }
}

extern "sysv64" fn entry_b(init: u64, run: u64) {
    log('B', init);
    log('b', run);
    let out = galloc(8);
    assert_eq!(f2("sceFiberReturnToThread")(3, out), 0);
    log('y', unsafe { *(out as *const u64) });
    assert_eq!(f3("sceFiberSwitch")(FIBER_A.load(Ordering::SeqCst), 5, 0), 0);
    loop {
        f2("sceFiberReturnToThread")(98, 0);
    }
}

/// Calls `stub(rdi, rsi, rdx)` with known values in every callee-saved
/// register and returns (rax, mask of registers that changed).
fn call_checked(stub: u64, a: u64, b: u64, c: u64) -> (i32, u64) {
    let rax: u64;
    let bad: u64;
    unsafe {
        std::arch::asm!(
            "push rbx",
            "push rbp",
            "sub rsp, 8",
            "mov rbx, 0x1111111111111111",
            "mov rbp, 0x2222222222222222",
            "mov r12, 0x3333333333333333",
            "mov r13, 0x4444444444444444",
            "mov r14, 0x5555555555555555",
            "mov r15, 0x6666666666666666",
            "call r11",
            "xor ecx, ecx",
            "mov rdx, 0x1111111111111111", "cmp rbx, rdx", "setne dl", "movzx edx, dl", "or rcx, rdx",
            "mov rdx, 0x2222222222222222", "cmp rbp, rdx", "setne dl", "movzx edx, dl", "shl edx, 1", "or rcx, rdx",
            "mov rdx, 0x3333333333333333", "cmp r12, rdx", "setne dl", "movzx edx, dl", "shl edx, 2", "or rcx, rdx",
            "mov rdx, 0x4444444444444444", "cmp r13, rdx", "setne dl", "movzx edx, dl", "shl edx, 3", "or rcx, rdx",
            "mov rdx, 0x5555555555555555", "cmp r14, rdx", "setne dl", "movzx edx, dl", "shl edx, 4", "or rcx, rdx",
            "mov rdx, 0x6666666666666666", "cmp r15, rdx", "setne dl", "movzx edx, dl", "shl edx, 5", "or rcx, rdx",
            "add rsp, 8",
            "pop rbp",
            "pop rbx",
            in("r11") stub,
            inout("rdi") a => _,
            inout("rsi") b => _,
            inout("rdx") c => _,
            out("rax") rax,
            out("rcx") bad,
            out("r12") _, out("r13") _, out("r14") _, out("r15") _,
            clobber_abi("sysv64"),
        );
    }
    (rax as i32, bad)
}

extern "sysv64" fn round_trip_thread(_arg: u64) -> u64 {
    FIBER_A.store(init_fiber("fiberA", entry_a, 100, 0x8000), Ordering::SeqCst);
    FIBER_B.store(init_fiber("fiberB", entry_b, 200, 0x8000), Ordering::SeqCst);
    let out = galloc(8);
    let a = FIBER_A.load(Ordering::SeqCst);
    let b = FIBER_B.load(Ordering::SeqCst);
    let (r, bad) = call_checked(stub("sceFiberRun"), a, 1, out);
    assert_eq!((r, bad), (0, 0), "first run");
    log('r', unsafe { *(out as *const u64) });
    // Not inside a fiber any more.
    assert_eq!(f3("sceFiberSwitch")(a, 0, 0), ps5hle::fiber::ERR_PERMISSION as i32);
    let (r, bad) = call_checked(stub("sceFiberRun"), b, 4, out);
    assert_eq!((r, bad), (0, 0), "second run");
    log('r', unsafe { *(out as *const u64) });
    let (r, bad) = call_checked(stub("sceFiberRun"), a, 7, out);
    assert_eq!((r, bad), (0, 0), "third run");
    log('r', unsafe { *(out as *const u64) });
    0x600D
}

#[test]
fn switch_round_trips() {
    let _g = SERIAL.lock().unwrap();
    stubs();
    assert_eq!(guest_thread(0, round_trip_thread, 0), 0x600D);
    let log = LOG.lock().unwrap().clone();
    let expect = vec![
        ('A', 100), ('a', 1), ('s', 1), // A starts with its args
        ('B', 200), ('b', 2), // A -> B (fresh)
        ('r', 3), // B -> thread
        ('y', 4), // thread -> B (resumed)
        ('x', 5), // B -> A (resumed)
        ('r', 6), // A -> thread
        ('z', 7), // thread -> A
        ('r', 99),
    ];
    assert_eq!(log, expect);
}

// ---------------------------------------------------------------------------
// Migration: one fiber resumed alternately by two threads.
// ---------------------------------------------------------------------------

static SHARED: AtomicU64 = AtomicU64::new(0);
static COUNT: AtomicU64 = AtomicU64::new(0);
static SERIAL: Mutex<()> = Mutex::new(());
const ROUNDS: u64 = 2000;

extern "sysv64" fn counter_entry(_init: u64, _run: u64) {
    let out = galloc(8);
    loop {
        COUNT.fetch_add(1, Ordering::SeqCst);
        // Burn a little stack to catch a stack shared by two threads.
        let mut buf = [0u64; 64];
        for (i, b) in buf.iter_mut().enumerate() {
            *b = i as u64;
        }
        std::hint::black_box(&buf);
        assert_eq!(f2("sceFiberReturnToThread")(buf[63], out), 0);
    }
}

extern "sysv64" fn migrate_thread(_arg: u64) -> u64 {
    let out = galloc(8);
    let fiber = SHARED.load(Ordering::SeqCst);
    let mut done = 0;
    while done < ROUNDS {
        let r = f3("sceFiberRun")(fiber, 0, out);
        if r == 0 {
            assert_eq!(unsafe { *(out as *const u64) }, 63);
            done += 1;
        } else {
            assert_eq!(r, ps5hle::fiber::ERR_STATE as i32);
            std::thread::yield_now();
        }
    }
    done
}

#[test]
fn fiber_migrates_between_threads() {
    let _g = SERIAL.lock().unwrap();
    stubs();
    extern "sysv64" fn setup(_arg: u64) -> u64 {
        SHARED.store(init_fiber("shared", counter_entry, 0, 0x8000), Ordering::SeqCst);
        0
    }
    guest_thread(3, setup, 0);
    let t1 = std::thread::spawn(|| guest_thread(1, migrate_thread, 0));
    let t2 = std::thread::spawn(|| guest_thread(2, migrate_thread, 0));
    assert_eq!(t1.join().unwrap() + t2.join().unwrap(), 2 * ROUNDS);
    assert_eq!(COUNT.load(Ordering::SeqCst), 2 * ROUNDS);
}

// ---------------------------------------------------------------------------
// A fresh fiber's frame chain ends at a zero record.
// ---------------------------------------------------------------------------

#[no_mangle]
static mut FRAME_PROBE: [u64; 3] = [1, 1, 1];

std::arch::global_asm!(
    ".globl frame_probe_entry",
    "frame_probe_entry:",
    "mov [rip + FRAME_PROBE], rbp",
    "mov rax, [rbp]",
    "mov [rip + FRAME_PROBE + 8], rax",
    "mov rax, [rbp + 8]",
    "mov [rip + FRAME_PROBE + 16], rax",
    "jmp {rest}",
    rest = sym frame_probe_rest,
);

extern "sysv64" {
    fn frame_probe_entry(init: u64, run: u64);
}

extern "sysv64" fn frame_probe_rest(_init: u64, _run: u64) {
    loop {
        f2("sceFiberReturnToThread")(0, 0);
    }
}

#[test]
fn fresh_fiber_frame_chain_ends_at_zero() {
    let _g = SERIAL.lock().unwrap();
    stubs();
    extern "sysv64" fn run(_arg: u64) -> u64 {
        // SAFETY: the same ABI and arguments; the entry is plain assembly.
        let entry: extern "sysv64" fn(u64, u64) = unsafe { std::mem::transmute(frame_probe_entry as unsafe extern "sysv64" fn(u64, u64)) };
        let fiber = init_fiber("frames", entry, 0, 0x8000);
        assert_eq!(f3("sceFiberRun")(fiber, 0, 0), 0);
        0
    }
    guest_thread(4, run, 0);
    // SAFETY: written once by the fiber entry, which has returned to the thread.
    let [rbp, saved, ret] = unsafe { std::ptr::addr_of!(FRAME_PROBE).read() };
    assert_ne!(rbp, 0);
    assert_eq!(rbp % 16, 0);
    assert_eq!((saved, ret), (0, 0));
}
