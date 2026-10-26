// RustPS5Emu (RPS5E)
// Copyright (C) 2025-2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Import dispatch: guest code calling import stubs through the trap
//! (INT3 + SIGTRAP) path and the direct entry path. Checks arguments,
//! results and preserved registers, tail calls, nested guest calls,
//! floating-point arguments, context replacement at the import-return safe
//! point, and reports the cost per call.

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use ps5cpu::exec::{self, GuestExit, HleDispatcher};
use ps5cpu::signal::{GuestInterrupt, InterruptKind};

const CODE: u64 = 0x5E10_0000_0000;
const DATA: u64 = 0x5E10_0000_1000;
const STACKS: u64 = 0x5E20_0000_0000;
const STACK_SIZE: u64 = 0x10_0000;

const LOOP: u64 = CODE;
const CALLER: u64 = CODE + 0x80;
const TRAP_STUB: u64 = CODE + 0x100;
const DIRECT_STUB: u64 = CODE + 0x110;
const TAIL_STUB: u64 = CODE + 0x120;
const NESTED_STUB: u64 = CODE + 0x130;
const XMM_STUB: u64 = CODE + 0x140;
const HOOK_STUB: u64 = CODE + 0x150;
const TRAP_TAIL_STUB: u64 = CODE + 0x160;
const RET_STUB: u64 = CODE + 0x170;
const TRAP_RET_STUB: u64 = CODE + 0x180;
const TRAP_ST0_STUB: u64 = CODE + 0x190;
const ST0_STUB: u64 = CODE + 0x1A0;
/// caller3(stub, arg): stub(arg); pop ST0 into [DATA+0x20] (10 bytes) and
/// store the x87 status word at [DATA+0x30].
const CALLER3: u64 = CODE + 0x380;
/// pi in x87 80-bit extended format.
const PI80: [u8; 10] = [0x35, 0xC2, 0x68, 0x21, 0xA2, 0xDA, 0x0F, 0xC9, 0x00, 0x40];
/// caller2(stub, arg): rax = stub(arg); [DATA+8] = rdx; [DATA+16] = xmm0.
const CALLER2: u64 = CODE + 0x300;
const ENTRY_SLOT: u64 = CODE + 0x200;
/// `lea rax, [rdi + 100]; ret`
const TAIL_TARGET: u64 = CODE + 0x220;
/// `lea rax, [rdi + rdi]; ret`
const NESTED_FN: u64 = CODE + 0x230;

static CALLS: AtomicU64 = AtomicU64::new(0);
thread_local! {
    /// Armed by HOOK_STUB for the import return of the same guest thread
    /// (the tests run in parallel: another thread's returns stay alone).
    static HOOK_ARMED: Cell<bool> = const { Cell::new(false) };
}

struct Dispatcher;

impl HleDispatcher for Dispatcher {
    fn dispatch_at(&self, stub: u64, regs: &[u64; 16], _rsp: u64, _rip: u64) -> Option<u64> {
        let rdi = regs[7];
        match stub {
            TRAP_STUB | DIRECT_STUB => {
                CALLS.fetch_add(1, Ordering::Relaxed);
                // rdi = remaining count; rsi = constant 0x1234 set by the loop.
                assert_eq!(regs[6], 0x1234);
                Some(rdi * 3)
            }
            TAIL_STUB | TRAP_TAIL_STUB => {
                exec::set_trap_tail_call(TAIL_TARGET);
                Some(0xDEAD)
            }
            NESTED_STUB => Some(exec::call_guest_nested(NESTED_FN, [rdi, 0, 0, 0, 0, 0]) + 1),
            XMM_STUB => exec::trap_xmm_low(0),
            RET_STUB | TRAP_RET_STUB => {
                exec::set_return_rdx(rdi + 1);
                exec::set_return_xmm0(1.5f64.to_bits() as u128);
                Some(rdi)
            }
            ST0_STUB | TRAP_ST0_STUB => {
                exec::set_return_st0(PI80);
                Some(rdi)
            }
            HOOK_STUB => {
                HOOK_ARMED.with(|a| a.set(true));
                Some(rdi)
            }
            _ => None,
        }
    }
}

/// At the import-return safe point of HOOK_STUB, replace r11 (a register the
/// fast return path does not restore) and r12.
fn hook(gi: &mut GuestInterrupt) {
    if gi.kind() != InterruptKind::ImportReturn || !HOOK_ARMED.with(|a| a.replace(false)) {
        return;
    }
    let mut uc = gi.ucontext();
    uc.uc_mcontext.mc_r11 = 0x1234_5678;
    uc.uc_mcontext.mc_rax += 1;
    assert!(gi.apply(&uc));
}

fn write(addr: u64, bytes: &[u8]) {
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), addr as *mut u8, bytes.len()) };
}

fn setup() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        exec::init();
        exec::set_dispatcher(Box::new(Dispatcher));
        ps5cpu::signal::set_signal_hook(hook);
        ps5vm::os::map_anon_noreplace(CODE, 0x2000, ps5vm::MemProtect::RWX).expect("map test code");
        // loop(count, stub): calls stub(count, 0x1234) `count` times, checks
        // that each call returned count*3 (traps with ud2 otherwise) and that
        // the callee-saved registers survived; returns the last result.
        #[rustfmt::skip]
        write(LOOP, &[
            0x41, 0x54,                         // push r12
            0x41, 0x55,                         // push r13
            0x53,                               // push rbx
            0x49, 0x89, 0xFC,                   // mov r12, rdi
            0x49, 0x89, 0xF5,                   // mov r13, rsi
            0x48, 0xC7, 0xC3, 0x55, 0x00, 0x00, 0x00, // mov rbx, 0x55
            // L:
            0x4C, 0x89, 0xE7,                   // mov rdi, r12
            0x48, 0xC7, 0xC6, 0x34, 0x12, 0x00, 0x00, // mov rsi, 0x1234
            0x41, 0xFF, 0xD5,                   // call r13
            0x4B, 0x8D, 0x0C, 0x64,             // lea rcx, [r12 + r12*2]
            0x48, 0x39, 0xC8,                   // cmp rax, rcx
            0x75, 0x11,                         // jne bad
            0x48, 0x83, 0xFB, 0x55,             // cmp rbx, 0x55
            0x75, 0x0B,                         // jne bad
            0x49, 0xFF, 0xCC,                   // dec r12
            0x75, 0xDF,                         // jnz L
            0x5B,                               // pop rbx
            0x41, 0x5D,                         // pop r13
            0x41, 0x5C,                         // pop r12
            0xC3,                               // ret
            // bad:
            0x0F, 0x0B,                         // ud2
        ]);
        // caller(stub, arg): xmm0 = arg; r11 = 0; rax = stub(arg);
        // [DATA] = r11 after the call; return rax.
        let mut caller = vec![
            0x53, // push rbx
            0x48, 0x89, 0xF8, // mov rax, rdi
            0x48, 0x89, 0xF7, // mov rdi, rsi
            0x66, 0x48, 0x0F, 0x6E, 0xC6, // movq xmm0, rsi
            0x45, 0x31, 0xDB, // xor r11d, r11d
            0xFF, 0xD0, // call rax
            0x50, // push rax
            0x4C, 0x89, 0xD8, // mov rax, r11
            0x48, 0xA3, // movabs [DATA], rax
        ];
        caller.extend_from_slice(&DATA.to_le_bytes());
        caller.extend_from_slice(&[0x58, 0x5B, 0xC3]); // pop rax; pop rbx; ret
        write(CALLER, &caller);
        let mut caller2 = vec![0x53, 0x48, 0x89, 0xF8, 0x48, 0x89, 0xF7, 0xFF, 0xD0, 0x48, 0x89, 0xC3, 0x48, 0x89, 0xD0, 0x48, 0xA3];
        caller2.extend_from_slice(&(DATA + 8).to_le_bytes());
        caller2.extend_from_slice(&[0x66, 0x48, 0x0F, 0x7E, 0xC0, 0x48, 0xA3]);
        caller2.extend_from_slice(&(DATA + 16).to_le_bytes());
        caller2.extend_from_slice(&[0x48, 0x89, 0xD8, 0x5B, 0xC3]);
        write(CALLER2, &caller2);
        write(TRAP_RET_STUB, &[0xCC]);
        write(TRAP_ST0_STUB, &[0xCC]);
        let mut caller3 = vec![0x53, 0x48, 0x89, 0xF8, 0x48, 0x89, 0xF7, 0xFF, 0xD0, 0x48, 0xB9]; // push rbx; mov rax, rdi; mov rdi, rsi; call rax; movabs rcx,
        caller3.extend_from_slice(&(DATA + 0x20).to_le_bytes());
        caller3.extend_from_slice(&[0xDB, 0x39, 0xDD, 0x79, 0x10, 0x5B, 0xC3]); // fstp tbyte [rcx]; fnstsw [rcx+0x10]; pop rbx; ret
        write(CALLER3, &caller3);
        write(TAIL_TARGET, &[0x48, 0x8D, 0x47, 0x64, 0xC3]);
        write(NESTED_FN, &[0x48, 0x8D, 0x04, 0x3F, 0xC3]);
        write(TRAP_STUB, &[0xCC]);
        write(TRAP_TAIL_STUB, &[0xCC]);
        unsafe { std::ptr::write(ENTRY_SLOT as *mut u64, exec::import_entry_address()) };
        for stub in [DIRECT_STUB, TAIL_STUB, NESTED_STUB, XMM_STUB, HOOK_STUB, RET_STUB, ST0_STUB] {
            write(stub, &exec::import_stub_code(stub, ENTRY_SLOT));
        }
        exec::add_guest_code_range(CODE, 0x1000, "import-dispatch-test");
        for i in 0..8 {
            exec::map_guest_stack(STACKS + (i + 1) * 0x1000_0000, STACK_SIZE).unwrap();
        }
    });
}

/// Run guest `func(args)` on its own guest thread with stack `slot`.
fn run_guest(slot: u64, func: u64, args: [u64; 6]) -> u64 {
    setup();
    let top = STACKS + (slot + 1) * 0x1000_0000;
    let h = std::thread::spawn(move || exec::enter_guest(&exec::call_regs(func, top, args)));
    match h.join().unwrap() {
        GuestExit::Returned(v) => v,
        e => panic!("unexpected exit {e:?}"),
    }
}

/// `count` calls through `stub`; returns ns per call.
fn bench(slot: u64, stub: u64, count: u64) -> f64 {
    let before = CALLS.load(Ordering::SeqCst);
    let start = Instant::now();
    assert_eq!(run_guest(slot, LOOP, [count, stub, 0, 0, 0, 0]), 3);
    let ns = start.elapsed().as_nanos() as f64 / count as f64;
    assert!(CALLS.load(Ordering::SeqCst) - before >= count);
    ns
}

#[test]
fn trap_and_direct_dispatch() {
    let trap = bench(0, TRAP_STUB, 20_000);
    let direct = bench(1, DIRECT_STUB, 2_000_000);
    eprintln!("import dispatch: trap {trap:.0} ns/call, direct {direct:.1} ns/call");
}

/// The callers share one result slot, so these run in sequence.
#[test]
fn direct_entry_semantics() {
    // Tail call: the import continues at TAIL_TARGET with the caller's
    // registers (rax = rdi + 100), then returns to the caller.
    assert_eq!(run_guest(2, CALLER, [TAIL_STUB, 5, 0, 0, 0, 0]), 105);
    assert_eq!(run_guest(2, CALLER, [TRAP_TAIL_STUB, 7, 0, 0, 0, 0]), 107);
    // Nested guest call made by the import.
    assert_eq!(run_guest(3, CALLER, [NESTED_STUB, 21, 0, 0, 0, 0]), 43);
    // Floating-point arguments.
    let bits = 2.5f64.to_bits();
    assert_eq!(run_guest(4, CALLER, [XMM_STUB, bits, 0, 0, 0, 0]), bits);
    // The hook replaces r11 and rax at the safe point: the direct entry
    // must install the whole context.
    assert_eq!(run_guest(5, CALLER, [HOOK_STUB, 41, 0, 0, 0, 0]), 42);
    assert_eq!(unsafe { std::ptr::read_volatile(DATA as *const u64) }, 0x1234_5678);
    // Extra return registers: RDX (128-bit results) and XMM0 (floating point).
    for stub in [RET_STUB, TRAP_RET_STUB] {
        unsafe { std::ptr::write_bytes((DATA + 8) as *mut u8, 0, 16) };
        assert_eq!(run_guest(6, CALLER2, [stub, 9, 0, 0, 0, 0]), 9);
        assert_eq!(unsafe { std::ptr::read_volatile((DATA + 8) as *const u64) }, 10);
        assert_eq!(unsafe { std::ptr::read_volatile((DATA + 16) as *const u64) }, 1.5f64.to_bits());
    }
    // `long double` results arrive in ST0 and leave the x87 stack balanced
    // once the caller pops them.
    for stub in [ST0_STUB, TRAP_ST0_STUB] {
        unsafe { std::ptr::write_bytes((DATA + 0x20) as *mut u8, 0xEE, 0x12) };
        assert_eq!(run_guest(7, CALLER3, [stub, 3, 0, 0, 0, 0]), 3);
        let st0 = unsafe { std::ptr::read_volatile((DATA + 0x20) as *const [u8; 10]) };
        assert_eq!(st0, PI80);
        let fsw = unsafe { std::ptr::read_volatile((DATA + 0x30) as *const u16) };
        assert_eq!((fsw >> 11) & 7, 0, "x87 TOP back at the initial slot");
        assert_eq!(fsw & 0x41, 0, "no stack fault or invalid operation");
    }
}

/// Pushing onto a saved FXSAVE image moves ST0..ST6 down, decrements TOP
/// and tags the new physical top register.
#[test]
fn saved_st0_push() {
    let mut fx = [0u8; 512];
    fx[2..4].copy_from_slice(&(3u16 << 11).to_le_bytes());
    fx[4] = 1 << 3;
    fx[32..42].copy_from_slice(&[1; 10]);
    exec::push_saved_st0(&mut fx, &PI80);
    assert_eq!(&fx[32..42], &PI80);
    assert_eq!(&fx[48..58], &[1; 10]);
    assert_eq!((u16::from_le_bytes([fx[2], fx[3]]) >> 11) & 7, 2);
    assert_eq!(fx[4], (1 << 3) | (1 << 2));
    // TOP wraps from 0 to 7.
    let mut fx = [0u8; 512];
    exec::push_saved_st0(&mut fx, &PI80);
    assert_eq!((u16::from_le_bytes([fx[2], fx[3]]) >> 11) & 7, 7);
    assert_eq!(fx[4], 1 << 7);
}
