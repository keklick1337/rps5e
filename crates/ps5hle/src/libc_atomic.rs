// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Dinkumware's sequentially consistent bitwise fetch helpers of
//! libSceLibcInternal: `_Fetch_{and,or,xor}_seq_cst_{1,2,4}(target, value)`
//! apply the operation atomically to the unsigned 8-, 16- or 32-bit object
//! at `target` and return its previous value.
//!
//! Used only when no loaded module exports them. The guest object lives at
//! the same host address, so the operations are the host's locked
//! read-modify-write instructions on it. A target that is not mapped
//! writable guest memory or not aligned to its size faults on the console;
//! here it is logged and nothing is changed.

use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};
use std::sync::atomic::{AtomicU16, AtomicU32, AtomicU8, Ordering};

const LIB: &str = "libSceLibcInternal";

#[derive(Clone, Copy)]
enum Op {
    And,
    Or,
    Xor,
}

/// Apply `op` with `value` to the `size`-byte object at `target`; the
/// previous value, or None for an unusable target.
fn fetch(target: u64, size: u64, op: Op, value: u64) -> Option<u64> {
    if target == 0 || target % size != 0 || !ps5vm::ranges::accessible(target, size, true) {
        log::error!("[libc] _Fetch_*_seq_cst_{size} on {target:#x}: not writable aligned guest memory");
        return None;
    }
    let order = Ordering::SeqCst;
    // SAFETY: the aligned object lies in mapped, writable guest memory,
    // which stays mapped while the guest uses it.
    unsafe {
        Some(match size {
            1 => {
                let a = &*(target as *const AtomicU8);
                let v = value as u8;
                (match op {
                    Op::And => a.fetch_and(v, order),
                    Op::Or => a.fetch_or(v, order),
                    Op::Xor => a.fetch_xor(v, order),
                }) as u64
            }
            2 => {
                let a = &*(target as *const AtomicU16);
                let v = value as u16;
                (match op {
                    Op::And => a.fetch_and(v, order),
                    Op::Or => a.fetch_or(v, order),
                    Op::Xor => a.fetch_xor(v, order),
                }) as u64
            }
            _ => {
                let a = &*(target as *const AtomicU32);
                let v = value as u32;
                (match op {
                    Op::And => a.fetch_and(v, order),
                    Op::Or => a.fetch_or(v, order),
                    Op::Xor => a.fetch_xor(v, order),
                }) as u64
            }
        })
    }
}

macro_rules! fetch_fn {
    ($name:ident, $op:expr, $size:expr) => {
        fn $name(ctx: &mut CpuContext) -> HleResult {
            fetch(ctx.rdi(), $size, $op, ctx.rsi()).unwrap_or(0) as HleResult
        }
    };
}

fetch_fn!(cb_fetch_and_1, Op::And, 1);
fetch_fn!(cb_fetch_and_2, Op::And, 2);
fetch_fn!(cb_fetch_and_4, Op::And, 4);
fetch_fn!(cb_fetch_or_1, Op::Or, 1);
fetch_fn!(cb_fetch_or_2, Op::Or, 2);
fetch_fn!(cb_fetch_or_4, Op::Or, 4);
fetch_fn!(cb_fetch_xor_1, Op::Xor, 1);
fetch_fn!(cb_fetch_xor_2, Op::Xor, 2);
fetch_fn!(cb_fetch_xor_4, Op::Xor, 4);

const FALLBACK_EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("zQQIrnpCoFA", "_Fetch_and_seq_cst_1", cb_fetch_and_1),
    ("xROUuk7ItMM", "_Fetch_and_seq_cst_2", cb_fetch_and_2),
    ("jQuruQuMlyo", "_Fetch_and_seq_cst_4", cb_fetch_and_4),
    ("2+6K-2tWaok", "_Fetch_or_seq_cst_1", cb_fetch_or_1),
    ("-egu08GJ0lw", "_Fetch_or_seq_cst_2", cb_fetch_or_2),
    ("gI9un1H-fZk", "_Fetch_or_seq_cst_4", cb_fetch_or_4),
    ("E2YhT7m79kM", "_Fetch_xor_seq_cst_1", cb_fetch_xor_1),
    ("fgXJvOSrqfg", "_Fetch_xor_seq_cst_2", cb_fetch_xor_2),
    ("cqcY17uV3dI", "_Fetch_xor_seq_cst_4", cb_fetch_xor_4),
];

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, cb) in FALLBACK_EXPORTS {
        reg.register_fallback_fn(nid, name, LIB, cb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CpuRegister;
    use ps5vm::{MemProtect, VirtualMemory};

    const BASE: u64 = 0x5E7C_0000_0000;

    #[test]
    fn nids_match_names() {
        for &(nid, name, _) in FALLBACK_EXPORTS {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    fn export(name: &str) -> HleCallback {
        FALLBACK_EXPORTS.iter().find(|e| e.1 == name).unwrap().2
    }

    fn expected(op: &str, before: u64, operand: u64) -> u64 {
        match op {
            "and" => before & operand,
            "or" => before | operand,
            _ => before ^ operand,
        }
    }

    /// Every helper returns the previous value, stores the bitwise result
    /// and leaves the neighbouring bytes alone; concurrent updates are not
    /// lost; a null or misaligned target changes nothing.
    #[test]
    fn fetch_helpers_return_the_previous_value() {
        let vm: &'static VirtualMemory = Box::leak(Box::new(VirtualMemory::new()));
        vm.map(BASE, 0x1000, MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(vm);
        let read = |ctx: &CpuContext, size: u64| -> u64 {
            let b = ctx.read_bytes(BASE + 8, size as usize).unwrap();
            b.iter().rev().fold(0, |v, &x| (v << 8) | x as u64)
        };
        for op in ["and", "or", "xor"] {
            for size in [1u64, 2, 4] {
                let f = export(&format!("_Fetch_{op}_seq_cst_{size}"));
                let max = if size == 4 { u32::MAX as u64 } else { (1u64 << (size * 8)) - 1 };
                let values = [0, 1, max, max >> 1, 1 << (size * 8 - 1), 0xA55A_A55A & max];
                for before in values {
                    for operand in values {
                        ctx.write_bytes(BASE, &[0xFF; 24]);
                        ctx.write_bytes(BASE + 8, &before.to_le_bytes()[..size as usize]);
                        ctx.set(CpuRegister::Rdi, BASE + 8);
                        ctx.set(CpuRegister::Rsi, operand | !max);
                        assert_eq!(f(&mut ctx) as u64, before, "{op}{size}");
                        assert_eq!(read(&ctx, size), expected(op, before, operand), "{op}{size}");
                        let guard = ctx.read_bytes(BASE, 24).unwrap();
                        assert!(guard[..8].iter().chain(&guard[8 + size as usize..]).all(|&b| b == 0xFF), "{op}{size}");
                    }
                }
                // Four threads each flip their own bit 1025 times.
                let start = if op == "and" { max } else { 0 };
                ctx.write_bytes(BASE + 8, &start.to_le_bytes()[..size as usize]);
                let workers: Vec<_> = (0..4u64)
                    .map(|bit| {
                        let operand = if op == "and" { !(1u64 << bit) & max } else { 1 << bit };
                        std::thread::spawn(move || {
                            let mut ctx = CpuContext::new(vm);
                            for _ in 0..1025 {
                                ctx.set(CpuRegister::Rdi, BASE + 8);
                                ctx.set(CpuRegister::Rsi, operand);
                                f(&mut ctx);
                            }
                        })
                    })
                    .collect();
                for w in workers {
                    w.join().unwrap();
                }
                let want = if op == "and" { max & !15 } else { 15 };
                assert_eq!(read(&ctx, size), want, "{op}{size} concurrent");
            }
        }
        ctx.write_bytes(BASE + 8, &[0x0F, 0, 0, 0]);
        for (target, size) in [(0u64, 1u64), (BASE + 9, 2), (BASE + 10, 4)] {
            let f = export(&format!("_Fetch_or_seq_cst_{size}"));
            ctx.set(CpuRegister::Rdi, target);
            ctx.set(CpuRegister::Rsi, 0xF0);
            assert_eq!(f(&mut ctx), 0);
        }
        assert_eq!(read(&ctx, 4), 0x0F);
    }
}
