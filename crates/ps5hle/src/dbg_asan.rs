// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! DbgAddressSanitizer — runtime support for AddressSanitizer builds.
//!
//! Instrumented code checks shadow memory at `(addr >> 3) + SHADOW_OFFSET`
//! before each access. The runtime here maps zeroed shadow pages (meaning
//! "addressable") for the initial stack, for registered globals and, once
//! initialised, for kernel memory allocations; no poisoning is performed.
//! Shadow pages are 64 KiB and reference counted.

use std::sync::Mutex;

use crate::context::{CpuContext, CpuRegister};
use crate::registry::{HleCallback, HleRegistry, HleResult};

const LIB: &str = "DbgAddressSanitizer";

const SHADOW_OFFSET: u64 = 0x100_0000_0000;
const SHADOW_PAGE: u64 = 0x1_0000;

/// One ASan global descriptor: { beg, size, size_with_redzone, name,
/// module_name, has_dynamic_init, location, odr_indicator }.
const GLOBAL_SIZE: u64 = 64;

/// Shadow pages by start address with their reference counts; None until
/// `__asan_init`.
static SHADOWS: Mutex<Option<Vec<(u64, u32)>>> = Mutex::new(None);

fn shadow_range(min_addr: u64, max_addr: u64) -> (u64, u64) {
    let lo = ((min_addr >> 3) + SHADOW_OFFSET) & !(SHADOW_PAGE - 1);
    let hi = ((max_addr >> 3) + SHADOW_OFFSET) & !(SHADOW_PAGE - 1);
    (lo, hi)
}

fn alloc_shadow(vm: &ps5vm::VirtualMemory, min_addr: u64, max_addr: u64) {
    let mut g = SHADOWS.lock().unwrap_or_else(|p| p.into_inner());
    let Some(shadows) = g.as_mut() else { return };
    let (lo, hi) = shadow_range(min_addr, max_addr);
    log::debug!("[asan] shadow {lo:#x} size {:#x}", hi - lo + SHADOW_PAGE);
    let mut page = lo;
    while page <= hi {
        if let Some(s) = shadows.iter_mut().find(|s| s.0 == page) {
            s.1 += 1;
        } else {
            if ps5vm::ranges::covered(page, SHADOW_PAGE) || vm.map(page, SHADOW_PAGE, ps5vm::MemProtect::RW).is_err() {
                log::error!("[asan] cannot allocate shadow memory at {page:#x}");
                ps5vm::shutdown::exit(1);
            }
            shadows.push((page, 1));
        }
        page += SHADOW_PAGE;
    }
}

fn free_shadow(vm: &ps5vm::VirtualMemory, min_addr: u64, max_addr: u64) {
    let mut g = SHADOWS.lock().unwrap_or_else(|p| p.into_inner());
    let Some(shadows) = g.as_mut() else { return };
    let (lo, hi) = shadow_range(min_addr, max_addr);
    let mut page = lo;
    while page <= hi {
        if let Some(i) = shadows.iter().position(|s| s.0 == page) {
            shadows[i].1 -= 1;
            if shadows[i].1 == 0 {
                if vm.unmap(page, SHADOW_PAGE).is_err() {
                    log::error!("[asan] cannot free shadow memory at {page:#x}");
                    ps5vm::shutdown::exit(1);
                }
                shadows.remove(i);
            }
        }
        page += SHADOW_PAGE;
    }
}

/// Kernel memory mapped for the guest: give it shadow once ASan runs.
pub fn kernel_alloc(vm: &ps5vm::VirtualMemory, addr: u64, size: u64) {
    if size != 0 {
        alloc_shadow(vm, addr, addr + size - 1);
    }
}

/// Kernel memory released by the guest.
pub fn kernel_free(vm: &ps5vm::VirtualMemory, addr: u64, size: u64) {
    if size != 0 {
        free_shadow(vm, addr, addr + size - 1);
    }
}

/// Lowest address of the host mapping that holds `addr` (the stack
/// reservation of the calling thread when `addr` is its stack pointer).
fn mapping_start(addr: u64) -> Option<u64> {
    let maps = std::fs::read_to_string("/proc/self/maps").ok()?;
    maps.lines().find_map(|l| {
        let range = l.split_whitespace().next()?;
        let (a, b) = range.split_once('-')?;
        let (a, b) = (u64::from_str_radix(a, 16).ok()?, u64::from_str_radix(b, 16).ok()?);
        (a <= addr && addr < b).then_some(a)
    })
}

/// __asan_init(): set up the runtime and shadow the whole current stack.
fn cb_asan_init(ctx: &mut CpuContext) -> HleResult {
    {
        let mut g = SHADOWS.lock().unwrap_or_else(|p| p.into_inner());
        if g.is_some() {
            log::error!("[asan] __asan_init called twice");
            ps5vm::shutdown::exit(1);
        }
        *g = Some(Vec::new());
    }
    let rsp = ctx.get(CpuRegister::Rsp);
    let base = mapping_start(rsp).unwrap_or(rsp);
    alloc_shadow(ctx.vm, base, rsp);
    0
}

fn cb_asan_version_mismatch_check_v8(_ctx: &mut CpuContext) -> HleResult {
    0
}

/// __asan_register_elf_globals(uptr* flag, Global* start, Global* stop)
fn cb_asan_register_elf_globals(ctx: &mut CpuContext) -> HleResult {
    let (flag, start, stop) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    if flag == 0 || start == 0 || stop == 0 || ctx.read_u64(flag) != Some(0) {
        return 0;
    }
    let num = stop.saturating_sub(start) / GLOBAL_SIZE;
    if num != 0 {
        let (mut min_addr, mut max_addr) = (u64::MAX, 0u64);
        for i in 0..num {
            let g = start + i * GLOBAL_SIZE;
            let beg = ctx.read_u64(g).unwrap_or(0);
            let size_with_redzone = ctx.read_u64(g + 16).unwrap_or(0);
            min_addr = min_addr.min(beg);
            max_addr = max_addr.max((beg + size_with_redzone).wrapping_sub(1));
        }
        alloc_shadow(ctx.vm, min_addr, max_addr);
    }
    ctx.write_u64(flag, 0);
    0
}

/// __asan_before_dynamic_init(const char* module_name)
fn cb_asan_before_dynamic_init(ctx: &mut CpuContext) -> HleResult {
    log::debug!("[asan] before dynamic init: {}", ctx.read_cstring(ctx.rdi(), 1024).unwrap_or_default());
    0
}

fn cb_asan_after_dynamic_init(_ctx: &mut CpuContext) -> HleResult {
    0
}

/// __asan_memcpy(dst, src, size) -> dst
fn cb_asan_memcpy(ctx: &mut CpuContext) -> HleResult {
    let (dst, src, n) = (ctx.rdi(), ctx.rsi(), ctx.rdx() as usize);
    if n != 0 {
        if let Some(b) = ctx.read_bytes(src, n) {
            ctx.write_bytes(dst, &b);
        }
    }
    dst as HleResult
}

/// __asan_memset(dst, ch, size) -> dst
fn cb_asan_memset(ctx: &mut CpuContext) -> HleResult {
    let (dst, ch, n) = (ctx.rdi(), ctx.rsi() as u8, ctx.rdx() as usize);
    if n != 0 {
        ctx.write_bytes(dst, &vec![ch; n]);
    }
    dst as HleResult
}

const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("xLoVJKHvRMU", "__asan_init", cb_asan_init),
    ("GY2I-Okc7q0", "__asan_version_mismatch_check_v8", cb_asan_version_mismatch_check_v8),
    ("W+2HQAGdPuU", "__asan_register_elf_globals", cb_asan_register_elf_globals),
    ("G9xSYlaNv+Y", "__asan_before_dynamic_init", cb_asan_before_dynamic_init),
    ("z5Gy5T3tptE", "__asan_after_dynamic_init", cb_asan_after_dynamic_init),
    ("g+Neom2EHO8", "__asan_memcpy", cb_asan_memcpy),
    ("q-hZf47nHhQ", "__asan_memset", cb_asan_memset),
];

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, cb) in EXPORTS {
        reg.register_fn(nid, name, LIB, cb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CpuRegister::*;

    #[test]
    fn nids_match_names() {
        for &(nid, name, _) in EXPORTS {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    #[test]
    fn shadow_page_arithmetic() {
        assert_eq!(shadow_range(0, 7), (SHADOW_OFFSET, SHADOW_OFFSET));
        assert_eq!(shadow_range(0x8_0000, 0x10_0000), (SHADOW_OFFSET + 0x1_0000, SHADOW_OFFSET + 0x2_0000));
    }

    #[test]
    fn asan_memory_helpers() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5D54_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.write_bytes(BASE, b"abcd");
        for (r, v) in [(Rdi, BASE + 0x10), (Rsi, BASE), (Rdx, 4)] {
            ctx.set(r, v);
        }
        assert_eq!(cb_asan_memcpy(&mut ctx), (BASE + 0x10) as HleResult);
        assert_eq!(ctx.read_bytes(BASE + 0x10, 4).unwrap(), b"abcd");
        ctx.set(Rsi, b'z' as u64);
        assert_eq!(cb_asan_memset(&mut ctx), (BASE + 0x10) as HleResult);
        assert_eq!(ctx.read_bytes(BASE + 0x10, 4).unwrap(), b"zzzz");
        // A non-zero registration flag skips the globals.
        ctx.write_u64(BASE + 0x100, 1);
        for (r, v) in [(Rdi, BASE + 0x100), (Rsi, BASE + 0x200), (Rdx, BASE + 0x240)] {
            ctx.set(r, v);
        }
        assert_eq!(cb_asan_register_elf_globals(&mut ctx), 0);
        assert_eq!(ctx.read_u64(BASE + 0x100), Some(1));
    }
}
