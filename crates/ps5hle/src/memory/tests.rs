// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Virtual memory allocation behaviour: flexible, direct, reserved and
//! pooled mappings, queries, protection, batches and automatic memory.
//!
//! The memory manager is process-wide, so these tests run one at a time.

use super::*;
use crate::context::CpuRegister;
use ps5vm::{MemProtect, VirtualMemory};
use std::sync::MutexGuard;

const PAGE: u64 = PAGE_SIZE;
const PROT_CPU_RW: i32 = 0x02;
const PROT_CPU_RD: i32 = 0x01;
const MTYPE_C: i32 = 11;
const POOL_RESERVE_LEN: u64 = 0x20_0000;
const POOL_COMMIT_LEN: u64 = 0x1_0000;
const POOL_EXPAND_LEN: u64 = 0x40_0000;
const POOL_ALIGN: u64 = 0x1_0000;

pub(super) static SERIAL: Mutex<()> = Mutex::new(());

struct T {
    _g: MutexGuard<'static, ()>,
    vm: &'static VirtualMemory,
    base: u64,
}

const BASE: u64 = 0x5CC0_0000_0000;

impl T {
    fn new() -> Self {
        let g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        static VM: std::sync::OnceLock<&'static VirtualMemory> = std::sync::OnceLock::new();
        let vm = *VM.get_or_init(|| {
            let vm: &'static VirtualMemory = Box::leak(Box::new(VirtualMemory::new()));
            vm.map(BASE, 0x10000, MemProtect::RW).unwrap();
            vm
        });
        T { _g: g, vm, base: BASE }
    }

    fn ctx(&self) -> CpuContext<'static> {
        CpuContext::new(self.vm)
    }

    fn call(&self, cb: crate::registry::HleCallback, args: &[u64]) -> (HleResult, CpuContext<'static>) {
        let mut ctx = self.ctx();
        for (r, v) in [CpuRegister::Rdi, CpuRegister::Rsi, CpuRegister::Rdx, CpuRegister::Rcx, CpuRegister::R8, CpuRegister::R9].iter().zip(args) {
            ctx.set(*r, *v);
        }
        let r = cb(&mut ctx);
        (r, ctx)
    }

    fn name(&self, slot: u64, s: &str) -> u64 {
        let a = self.base + 0x8000 + slot * 0x100;
        let mut b = s.as_bytes().to_vec();
        b.push(0);
        self.ctx().write_bytes(a, &b);
        a
    }

    /// Map flexible memory: (result, address).
    fn flex(&self, addr: u64, len: u64, prot: i32, flags: u64, name: Option<&str>) -> (HleResult, u64) {
        let io = self.base;
        self.ctx().write_u64(io, addr);
        let n = name.map_or(0, |s| self.name(0, s));
        let (r, ctx) = self.call(cb_map_named_flexible_memory, &[io, len, prot as u64, flags, n]);
        (r, ctx.read_u64(io).unwrap())
    }

    fn flex_ok(&self, len: u64, prot: i32, name: &str) -> u64 {
        let (r, a) = self.flex(0, len, prot, 0, Some(name));
        assert_eq!(r, OK, "map flexible {name}");
        assert_ne!(a, 0);
        a
    }

    fn alloc(&self, start: u64, end: u64, len: u64, align: u64, mtype: i32) -> (HleResult, u64) {
        let out = self.base + 0x10;
        self.ctx().write_u64(out, u64::MAX);
        let (r, ctx) = self.call(cb_allocate_direct_memory, &[start, end, len, align, mtype as u64, out]);
        (r, ctx.read_u64(out).unwrap())
    }

    /// sceKernelAvailableDirectMemorySize: (result, phys, size).
    fn available_direct(&self, start: u64, end: u64, align: u64) -> (HleResult, u64, u64) {
        let (po, so) = (self.base + 0x90, self.base + 0x98);
        let (r, ctx) = self.call(cb_available_direct_memory_size, &[start, end, align, po, so]);
        (r, ctx.read_u64(po).unwrap(), ctx.read_u64(so).unwrap())
    }

    fn alloc_ok(&self, len: u64, align: u64) -> u64 {
        let (r, p) = self.alloc(0, direct_size(), len, align, MTYPE_C);
        assert_eq!(r, OK);
        p
    }

    /// Map named direct memory: (result, address).
    #[allow(clippy::too_many_arguments)]
    fn direct(&self, addr: u64, len: u64, prot: i32, flags: u64, phys: u64, align: u64, name: &str) -> (HleResult, u64) {
        let io = self.base + 0x20;
        let mut ctx = self.ctx();
        ctx.write_u64(io, addr);
        let n = self.name(1, name);
        // The name is the seventh argument, on the stack.
        let rsp = self.base + 0x400;
        ctx.write_u64(rsp + 8, n);
        for (r, v) in [CpuRegister::Rdi, CpuRegister::Rsi, CpuRegister::Rdx, CpuRegister::Rcx, CpuRegister::R8, CpuRegister::R9]
            .iter()
            .zip([io, len, prot as u64, flags, phys, align])
        {
            ctx.set(*r, v);
        }
        ctx.set(CpuRegister::Rsp, rsp);
        let r = cb_map_named_direct_memory(&mut ctx);
        (r, ctx.read_u64(io).unwrap())
    }

    fn reserve(&self, addr: u64, len: u64, flags: u64, align: u64) -> (HleResult, u64) {
        let io = self.base + 0x30;
        self.ctx().write_u64(io, addr);
        let (r, ctx) = self.call(cb_reserve_virtual_range, &[io, len, flags, align]);
        (r, ctx.read_u64(io).unwrap())
    }

    fn munmap(&self, addr: u64, len: u64) -> HleResult {
        unmap(addr, len)
    }

    fn release(&self, phys: u64, len: u64) -> HleResult {
        self.call(cb_release_direct_memory, &[phys, len]).0
    }

    fn checked_release(&self, phys: u64, len: u64) -> HleResult {
        self.call(cb_checked_release_direct_memory, &[phys, len]).0
    }

    fn available_flexible(&self) -> u64 {
        let out = self.base + 0x40;
        let (r, ctx) = self.call(cb_available_flexible_memory_size, &[out]);
        assert_eq!(r, OK);
        ctx.read_u64(out).unwrap()
    }

    fn query_raw(&self, addr: u64, flags: i32) -> (HleResult, [u8; 72]) {
        let info = self.base + 0x100;
        let (r, ctx) = self.call(cb_virtual_query, &[addr, flags as u64, info, 72]);
        let mut b = [0u8; 72];
        b.copy_from_slice(&ctx.read_bytes(info, 72).unwrap());
        (r, b)
    }

    fn query(&self, addr: u64, flags: i32) -> Q {
        let (r, b) = self.query_raw(addr, flags);
        assert_eq!(r, OK, "query {addr:#x}");
        Q::from(&b)
    }

    fn expect_unmapped(&self, addr: u64) {
        assert_eq!(self.query_raw(addr, 0).0, EACCES, "{addr:#x} should be unmapped");
    }

    fn write(&self, addr: u64, v: u64) {
        // SAFETY: test-owned guest mapping.
        unsafe { (addr as *mut u64).write_volatile(v) }
    }

    fn read(&self, addr: u64) -> u64 {
        // SAFETY: test-owned guest mapping.
        unsafe { (addr as *const u64).read_volatile() }
    }
}

#[derive(Debug)]
struct Q {
    start: u64,
    end: u64,
    offset: u64,
    prot: i32,
    mtype: i32,
    flexible: bool,
    direct: bool,
    stack: bool,
    pooled: bool,
    committed: bool,
    name: String,
}

impl Q {
    fn from(b: &[u8; 72]) -> Self {
        let u64at = |o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
        let bits = b[32];
        let name_end = b[33..65].iter().position(|&c| c == 0).unwrap_or(32);
        Q {
            start: u64at(0),
            end: u64at(8),
            offset: u64at(16),
            prot: i32::from_le_bytes(b[24..28].try_into().unwrap()),
            mtype: i32::from_le_bytes(b[28..32].try_into().unwrap()),
            flexible: bits & 1 != 0,
            direct: bits & 2 != 0,
            stack: bits & 4 != 0,
            pooled: bits & 8 != 0,
            committed: bits & 16 != 0,
            name: String::from_utf8_lossy(&b[33..33 + name_end]).into_owned(),
        }
    }

    /// (start, end, prot, flexible, direct, pooled, committed)
    fn shape(&self) -> (u64, u64, i32, bool, bool, bool, bool) {
        (self.start, self.end, self.prot, self.flexible, self.direct, self.pooled, self.committed)
    }
}

#[test]
fn argument_and_info_size_contracts() {
    let t = T::new();
    assert_eq!(t.flex(0, 0, PROT_CPU_RW, 0, Some("zero_len")).0, EINVAL);
    assert_eq!(t.query_raw(0, 0).0, EACCES);
    let info = t.base + 0x100;
    assert_eq!(t.call(cb_virtual_query, &[0, 0, info, 71]).0, EINVAL);
    assert_eq!(t.call(cb_virtual_query, &[0, 2, info, 72]).0, EINVAL);
    assert_eq!(t.flex(0, PAGE, PROT_CPU_RW, 0, None).0, EFAULT, "name required");
    assert_eq!(t.flex(0, PAGE, PROT_CPU_RW, 0, Some(&"x".repeat(32))).0, ENAMETOOLONG);
}

#[test]
fn fixed_mappings_need_a_guest_region() {
    let t = T::new();
    let (r, a) = t.reserve(0x10000, PAGE, MAP_FIXED | MAP_NO_OVERWRITE, PAGE);
    assert!(r < 0);
    assert_eq!(a, 0x10000, "a failed fixed reservation does not move");
    assert!(t.flex(0x10000, PAGE, PROT_CPU_RW, MAP_FIXED | MAP_NO_OVERWRITE, Some("unowned")).0 < 0);
    let phys = t.alloc_ok(PAGE, PAGE);
    assert!(t.direct(0x10000, PAGE, PROT_CPU_RW, MAP_FIXED | MAP_NO_OVERWRITE, phys, PAGE, "unowned").0 < 0);
    assert_eq!(t.checked_release(phys, PAGE), OK);
}

#[test]
fn flexible_capacity_is_fixed_at_boot() {
    let t = T::new();
    let out = t.base + 0x50;
    let (r, ctx) = t.call(cb_configured_flexible_memory_size, &[out]);
    assert_eq!(r, OK);
    let configured = ctx.read_u64(out).unwrap();
    assert_eq!(configured, flexible_size());
    assert_eq!(direct_size() + configured, TOTAL_BYTES, "direct and flexible partition the backing");
    let baseline = t.available_flexible();
    let a = t.flex_ok(PAGE, PROT_CPU_RW, "boot_fixed_flexible");
    assert_eq!(t.available_flexible(), baseline - PAGE);
    assert!(!set_flexible_memory_size(PAGE * 4), "capacity cannot change once running");
    assert_eq!(t.munmap(a, PAGE), OK);
    assert_eq!(t.available_flexible(), baseline);
}

#[test]
fn flexible_dmem_compat_and_alignment_flags() {
    let t = T::new();
    let baseline = t.available_flexible();
    let (r, a) = t.flex(0, PAGE, PROT_CPU_RW, MAP_DMEM_COMPAT | (16 << 24), Some("dmem_compat"));
    assert_eq!(r, OK);
    assert_eq!(a % 0x10000, 0, "alignment flag honoured");
    let q = t.query(a, 0);
    assert!(q.flexible && !q.stack);
    let (s, e) = (t.base + 0x60, t.base + 0x68);
    t.ctx().write_u64(s, u64::MAX);
    t.ctx().write_u64(e, u64::MAX);
    let (r, ctx) = t.call(cb_is_stack, &[a, s, e]);
    assert_eq!(r, OK);
    assert_eq!((ctx.read_u64(s), ctx.read_u64(e)), (Some(0), Some(0)), "not a stack");
    assert_eq!(t.munmap(a, PAGE), OK);
    let (r, a) = t.flex(0, PAGE, PROT_CPU_RW, 0x8000, Some("opaque_runtime_flag"));
    assert_eq!(r, OK);
    assert_eq!(t.available_flexible() + PAGE, baseline);
    assert_eq!(t.munmap(a, PAGE), OK);
    assert_eq!(t.available_flexible(), baseline);
    assert_eq!(t.flex(0, PAGE, PROT_CPU_RW, 0x10000, Some("unsupported")).0, EINVAL);
    assert_eq!(t.flex(0, PAGE, PROT_CPU_RW, 13 << 24, Some("bad_alignment")).0, EINVAL);
}

#[test]
fn flexible_no_coalesce_keeps_boundaries() {
    let t = T::new();
    let baseline = t.available_flexible();
    let (r, base) = t.reserve(0, PAGE * 2, 0, PAGE);
    assert_eq!(r, OK);
    assert_eq!(t.flex(base, PAGE, PROT_CPU_RW, MAP_FIXED | MAP_NO_COALESCE, Some("no_coalesce")).0, OK);
    assert_eq!(t.flex(base + PAGE, PAGE, PROT_CPU_RW, MAP_FIXED | MAP_NO_COALESCE, Some("no_coalesce")).0, OK);
    assert_eq!(t.query(base, 0).shape(), (base, base + PAGE, reported_protection(PROT_CPU_RW), true, false, false, true));
    assert_eq!(t.query(base + PAGE, 0).shape(), (base + PAGE, base + 2 * PAGE, reported_protection(PROT_CPU_RW), true, false, false, true));
    assert_eq!(t.query(base, 0).name, "no_coalesce");
    assert_eq!(t.munmap(base, PAGE * 2), OK);
    assert_eq!(t.available_flexible(), baseline);
}

#[test]
fn flexible_reuse_is_zero_filled() {
    let t = T::new();
    let baseline = t.available_flexible();
    let a = t.flex_ok(PAGE, PROT_CPU_RW, "zero_source");
    // SAFETY: freshly mapped flexible memory.
    unsafe { std::ptr::write_bytes(a as *mut u8, 0xa5, PAGE as usize) };
    assert_eq!(t.munmap(a, PAGE), OK);
    let b = t.flex_ok(PAGE, PROT_CPU_RW, "zero_reuse");
    // SAFETY: freshly mapped flexible memory.
    let bytes = unsafe { std::slice::from_raw_parts(b as *const u8, PAGE as usize) };
    assert!(bytes.iter().all(|&x| x == 0), "reused backing exposed stale bytes");
    assert_eq!(t.munmap(b, PAGE), OK);
    assert_eq!(t.available_flexible(), baseline);
}

#[test]
fn smaller_flexible_map_reuses_released_hole() {
    let t = T::new();
    let baseline = t.available_flexible();
    let (small, large, block) = (PAGE * 12, PAGE * 16, PAGE * 4);
    let (_, seed_hole) = t.flex(0, small, PROT_CPU_RW, 0x8000, Some("seed_hole"));
    let (_, seed_blocker) = t.flex(0, block, PROT_CPU_RW, 0x8000, Some("seed_blocker"));
    assert_eq!(t.munmap(seed_hole, small), OK);
    let (r, vblock) = t.reserve(seed_hole, small, 0, PAGE);
    assert_eq!((r, vblock), (OK, seed_hole), "hole reserved in place");
    let (r, l) = t.flex(0, large, PROT_CPU_RW, 0x8000, Some("large"));
    assert_eq!(r, OK);
    let (r, b) = t.flex(0, small, PROT_CPU_RW, 0x8000, Some("blocker"));
    assert_eq!(r, OK);
    assert_eq!(b, l + large, "hint-less maps are adjacent");
    assert_eq!(t.munmap(l, large), OK);
    let (r, reused) = t.flex(0, small, PROT_CPU_RW, 0x8000, Some("reuse"));
    assert_eq!((r, reused), (OK, l), "smaller map reuses the released hole");
    t.write(reused, 0x4b59_5459_5245_5553);
    assert_eq!(t.read(reused), 0x4b59_5459_5245_5553);
    for (a, s) in [(reused, small), (b, small), (seed_blocker, block), (vblock, small)] {
        assert_eq!(t.munmap(a, s), OK);
    }
    assert_eq!(t.available_flexible(), baseline);
}

#[test]
fn flexible_query_partial_munmap_and_find_next() {
    let t = T::new();
    let baseline = t.available_flexible();
    let base = t.flex_ok(PAGE * 2, PROT_CPU_RW, "prospero_flex");
    let q = t.query(base, 0);
    assert_eq!(q.shape(), (base, base + 2 * PAGE, reported_protection(PROT_CPU_RW), true, false, false, true));
    assert_eq!(q.name, "prospero_flex");
    assert_eq!(t.available_flexible() + 2 * PAGE, baseline);
    assert_eq!(t.munmap(base, PAGE * 2), OK);
    t.expect_unmapped(base);

    let base = t.flex_ok(PAGE * 3, PROT_CPU_RW, "prospero_part");
    assert_eq!(t.munmap(base + PAGE, PAGE), OK);
    assert_eq!(t.query(base, 0).shape(), (base, base + PAGE, reported_protection(PROT_CPU_RW), true, false, false, true));
    t.expect_unmapped(base + PAGE);
    assert_eq!(t.query(base + PAGE, 1).shape(), (base + 2 * PAGE, base + 3 * PAGE, reported_protection(PROT_CPU_RW), true, false, false, true));
    assert_eq!(t.available_flexible() + 2 * PAGE, baseline);
    assert_eq!(t.munmap(base, PAGE), OK);
    assert_eq!(t.munmap(base + 2 * PAGE, PAGE), OK);
    assert_eq!(t.available_flexible(), baseline);
}

#[test]
fn reserve_fixed_map_and_no_overwrite() {
    let t = T::new();
    let (r, base) = t.reserve(0, PAGE * 3, 0, PAGE);
    assert_eq!(r, OK);
    let q = t.query(base, 0);
    assert_eq!(q.shape(), (base, base + 3 * PAGE, 0, false, false, false, false));
    assert_eq!(q.name, "anon");
    let (r, fixed) = t.flex(base + PAGE, PAGE, PROT_CPU_RD, MAP_FIXED, Some("fixed_mid"));
    assert_eq!((r, fixed), (OK, base + PAGE));
    assert_eq!(t.query(base, 0).shape(), (base, base + PAGE, 0, false, false, false, false));
    assert_eq!(t.query(base + PAGE, 0).shape(), (base + PAGE, base + 2 * PAGE, PROT_CPU_RD, true, false, false, true));
    assert_eq!(t.query(base + 2 * PAGE, 0).shape(), (base + 2 * PAGE, base + 3 * PAGE, 0, false, false, false, false));
    assert_eq!(t.flex(base + PAGE, PAGE, PROT_CPU_RW, MAP_FIXED | MAP_NO_OVERWRITE, Some("blocked")).0, ENOMEM);
    assert_eq!(t.query(base + PAGE, 0).name, "fixed_mid");
    // An existing reservation also blocks MAP_NO_OVERWRITE.
    assert_eq!(t.flex(base, PAGE, PROT_CPU_RW, MAP_FIXED | MAP_NO_OVERWRITE, Some("reserved_blocked")).0, ENOMEM);
    for i in 0..3 {
        assert_eq!(t.munmap(base + i * PAGE, PAGE), OK);
    }
}

#[test]
fn released_reserve_can_be_reused() {
    let t = T::new();
    let (r, base) = t.reserve(0, PAGE, 0, PAGE);
    assert_eq!(r, OK);
    assert_eq!(t.munmap(base, PAGE), OK);
    let (r, again) = t.reserve(base, PAGE, MAP_FIXED | MAP_NO_OVERWRITE, PAGE);
    assert_eq!((r, again), (OK, base));
    assert_eq!(t.munmap(base, PAGE), OK);
}

#[test]
fn munmap_across_adjacent_flexible_mappings() {
    let t = T::new();
    let baseline = t.available_flexible();
    let (_, base) = t.reserve(0, PAGE * 2, 0, PAGE);
    assert_eq!(t.flex(base, PAGE, PROT_CPU_RW, MAP_FIXED, Some("adjacent_left")).0, OK);
    assert_eq!(t.flex(base + PAGE, PAGE, PROT_CPU_RW, MAP_FIXED, Some("adjacent_right")).0, OK);
    assert_eq!(clamp_range_size(base + PAGE - 0x100, 0x200), 0x200, "clamp crosses committed neighbours");
    assert_eq!(t.munmap(base, PAGE * 2), OK);
    assert_eq!(t.available_flexible(), baseline);
    t.expect_unmapped(base);
    t.expect_unmapped(base + PAGE);
}

#[test]
fn direct_query_offset_aliases_and_partial_munmap() {
    let t = T::new();
    let phys = t.alloc_ok(PAGE * 4, PAGE);
    let (r, base) = t.direct(0, PAGE * 4, PROT_CPU_RW, 0, phys, PAGE, "prospero_direct");
    assert_eq!(r, OK);
    let (r, alias) = t.direct(0, PAGE * 4, PROT_CPU_RW, 0, phys, PAGE, "prospero_direct_alias");
    assert_eq!(r, OK);
    t.write(base, 0x4b59_5459_444d_454d);
    assert_eq!(t.read(alias), 0x4b59_5459_444d_454d, "views of one physical range share bytes");
    let q = t.query(base, 0);
    assert_eq!(q.shape(), (base, base + 4 * PAGE, reported_protection(PROT_CPU_RW), false, true, false, true));
    assert_eq!((q.offset, q.mtype, q.name.as_str()), (phys, MTYPE_C, "prospero_direct"));
    assert_eq!(t.munmap(base + PAGE, PAGE), OK);
    t.expect_unmapped(base + PAGE);
    assert_eq!(clamp_range_size(base + PAGE - 0xf30, 0x1560), 0xf30, "clamp stops at a hole");
    let q = t.query(base + PAGE, 1);
    assert_eq!(q.shape(), (base + 2 * PAGE, base + 4 * PAGE, reported_protection(PROT_CPU_RW), false, true, false, true));
    assert_eq!((q.offset, q.mtype), (phys + 2 * PAGE, MTYPE_C));
    assert_eq!(t.munmap(base, PAGE), OK);
    assert_eq!(t.munmap(base + 2 * PAGE, 2 * PAGE), OK);
    assert_eq!(t.munmap(alias, 4 * PAGE), OK);
    assert_eq!(t.release(phys, 4 * PAGE), OK);
}

#[test]
fn direct_partial_protect_then_release_unmaps_everything() {
    let t = T::new();
    let size = PAGE * 3;
    let phys = t.alloc_ok(size, PAGE);
    let (r, base) = t.direct(0, size, PROT_CPU_RW, 0, phys, PAGE, "partial_protect_direct");
    assert_eq!(r, OK);
    assert_eq!(t.call(cb_mprotect, &[base + PAGE, PAGE, PROT_CPU_RD as u64]).0, OK);
    assert_eq!(t.query(base + PAGE, 0).prot, PROT_CPU_RD);
    assert_eq!(t.query(base, 0).prot, reported_protection(PROT_CPU_RW));
    assert_eq!(t.munmap(base + PAGE, PAGE), OK);
    t.write(base, 0x4c45_4654_5052_4f54);
    t.write(base + 2 * PAGE, 0x5247_4854_5052_4f54);
    assert_eq!(t.read(base), 0x4c45_4654_5052_4f54);
    assert_eq!(t.read(base + 2 * PAGE), 0x5247_4854_5052_4f54);
    assert_eq!(t.release(phys, size), OK);
    t.expect_unmapped(base);
    t.expect_unmapped(base + 2 * PAGE);
}

#[test]
fn direct_names_types_and_validation() {
    let t = T::new();
    let phys = t.alloc_ok(PAGE * 2, PAGE);
    // The name is validated first.
    assert_eq!(t.direct(0, 0, PROT_CPU_EXEC, 0, u64::MAX, 0, &"x".repeat(32)).0, ENAMETOOLONG);
    let (_, base) = t.reserve(0, PAGE * 2, 0, PAGE);
    let long = "x".repeat(31);
    assert_eq!(t.direct(base, PAGE, PROT_CPU_RW, MAP_FIXED, phys, PAGE, &long).0, OK);
    assert_eq!(t.direct(base + PAGE, PAGE, PROT_CPU_RW, MAP_FIXED, phys + PAGE, PAGE, "second").0, OK);
    let q = t.query(base, 0);
    assert_eq!((q.start, q.end, q.offset, q.name.as_str()), (base, base + PAGE, phys, long.as_str()));
    let q = t.query(base + PAGE, 0);
    assert_eq!((q.start, q.end, q.offset, q.name.as_str()), (base + PAGE, base + 2 * PAGE, phys + PAGE, "second"));
    // MapDirectMemory2 with an explicit type; -1 keeps the allocation's type.
    let io = t.base + 0x20;
    let rsp = t.base + 0x400;
    let map2 = |addr: u64, mtype: i64| {
        let mut ctx = t.ctx();
        ctx.write_u64(io, addr);
        ctx.write_u64(rsp + 8, PAGE);
        for (r, v) in [CpuRegister::Rdi, CpuRegister::Rsi, CpuRegister::Rdx, CpuRegister::Rcx, CpuRegister::R8, CpuRegister::R9]
            .iter()
            .zip([io, PAGE, mtype as u64, PROT_CPU_RW as u64, 0, phys])
        {
            ctx.set(*r, v);
        }
        ctx.set(CpuRegister::Rsp, rsp);
        let r = cb_map_direct_memory2(&mut ctx);
        (r, ctx.read_u64(io).unwrap())
    };
    let (r, typed) = map2(0, 2);
    assert_eq!(r, OK);
    assert_eq!(t.query(typed, 0).mtype, 2);
    assert_eq!(t.query(base, 0).mtype, MTYPE_C, "a typed alias keeps other views' types");
    t.write(typed, 0x1234_5678);
    assert_eq!(t.read(base), 0x1234_5678);
    let (r, unchanged) = map2(0, -1);
    assert_eq!(r, OK);
    assert_eq!(t.query(unchanged, 0).mtype, MTYPE_C);
    let (r, plain) = {
        let io = t.base + 0x28;
        t.ctx().write_u64(io, 0);
        let (r, ctx) = t.call(cb_map_direct_memory, &[io, PAGE, PROT_CPU_RW as u64, 0, phys, PAGE]);
        (r, ctx.read_u64(io).unwrap())
    };
    assert_eq!(r, OK);
    let q = t.query(plain, 0);
    assert_eq!((q.mtype, q.name.as_str()), (MTYPE_C, ""));
    assert_eq!(t.release(phys, PAGE * 2), OK);
    for a in [base, typed, plain, unchanged] {
        t.expect_unmapped(a);
    }
}

#[test]
fn direct_map_validation_leaves_outputs() {
    let t = T::new();
    let (r, p) = t.alloc(0, direct_size(), PAGE + 1, PAGE, MTYPE_C);
    assert_eq!((r, p), (EINVAL, u64::MAX), "invalid size leaves the output");
    let (r, p) = t.alloc(0, direct_size(), PAGE, 0x1000, MTYPE_C);
    assert_eq!((r, p), (EINVAL, u64::MAX), "sub-page alignment");
    let phys = t.alloc_ok(PAGE * 2, PAGE);
    let io = t.base + 0x28;
    let map = |len: u64, prot: i32, flags: u64, phys: u64, align: u64| {
        t.ctx().write_u64(io, 0);
        let (r, ctx) = t.call(cb_map_direct_memory, &[io, len, prot as u64, flags, phys, align]);
        (r, ctx.read_u64(io).unwrap())
    };
    assert_eq!(map(PAGE + 1, PROT_CPU_RW, 0, phys, PAGE), (EINVAL, 0));
    assert_eq!(map(PAGE, PROT_CPU_RW, 0, phys + 1, PAGE), (EINVAL, 0));
    assert_eq!(map(PAGE, PROT_CPU_EXEC, 0, phys, PAGE), (EACCES, 0));
    let (r, aligned) = map(PAGE, PROT_CPU_RW, 0, phys, 0xc000);
    assert_eq!(r, OK);
    assert_eq!(aligned % 0xc000, 0, "16 KiB-multiple alignment honoured");
    assert_eq!(t.munmap(aligned, PAGE), OK);
    let (r, ignored) = map(PAGE, PROT_CPU_RW, 0x08, phys, PAGE);
    assert_eq!(r, OK, "unknown flag bits are ignored");
    assert_eq!(t.munmap(ignored, PAGE), OK);
    assert_eq!(t.release(phys, PAGE * 2), OK);
}

#[test]
fn direct_release_contracts() {
    let t = T::new();
    assert_eq!(t.release(0, 0), OK);
    assert_eq!(t.checked_release(0, 0), OK);
    assert_eq!(t.release(1, PAGE), EINVAL);
    assert_eq!(t.release(0, PAGE + 1), EINVAL);
    let free = direct_size() - PAGE;
    assert_eq!(t.release(free, PAGE), OK, "unchecked release ignores unallocated ranges");
    assert_eq!(t.checked_release(free, PAGE), ENOENT);
}

#[test]
fn direct_allocation_reuse_coalescing_and_alignment() {
    let t = T::new();
    let end = direct_size();
    let first = t.alloc_ok(PAGE, PAGE);
    let second = t.alloc_ok(PAGE, PAGE);
    assert_eq!(second, first + PAGE, "contiguous allocations");
    let (r, a) = t.direct(0, PAGE, PROT_CPU_RW, 0, second, PAGE, "nonzero_a");
    assert_eq!(r, OK);
    let (r, b) = t.direct(0, PAGE, PROT_CPU_RW, 0, second, PAGE, "nonzero_b");
    assert_eq!(r, OK);
    t.write(a, 0x4b59_5459_3136_4b42);
    assert_eq!(t.read(b), 0x4b59_5459_3136_4b42);
    assert_eq!(t.munmap(a, PAGE), OK);
    assert_eq!(t.munmap(b, PAGE), OK);
    // A view across two contiguous allocations; a checked release of both
    // unmaps it and frees one coalesced span.
    let (r, span) = t.direct(0, PAGE * 2, PROT_CPU_RW, 0, first, PAGE, "contiguous");
    assert_eq!(r, OK);
    assert_eq!(t.checked_release(first, PAGE * 2), OK);
    t.expect_unmapped(span);
    let (r, reclaimed) = t.alloc(0, end, PAGE * 2, PAGE, MTYPE_C);
    assert_eq!((r, reclaimed), (OK, first));
    assert_eq!(t.checked_release(reclaimed, PAGE * 2), OK);

    let first = t.alloc_ok(PAGE * 3, PAGE);
    let (middle, last) = (first + PAGE, first + 2 * PAGE);
    assert_eq!(t.release(middle, PAGE), OK);
    let (_, reused) = t.alloc(0, end, PAGE, PAGE, MTYPE_C);
    assert_eq!(reused, middle, "released gap reused");
    for p in [first, reused, last] {
        assert_eq!(t.release(p, PAGE), OK);
    }
    let (_, coalesced) = t.alloc(0, end, PAGE * 3, PAGE, MTYPE_C);
    assert_eq!(coalesced, first, "adjacent free ranges coalesce");
    assert_eq!(t.release(coalesced, PAGE * 3), OK);

    let (r, p) = t.alloc(PAGE * 2, end, PAGE, PAGE * 3, MTYPE_C);
    assert_eq!(r, OK);
    assert!(p >= PAGE * 2 && p % (PAGE * 3) == 0, "non-power-of-two alignment within the window");
    assert_eq!(t.release(p, PAGE), OK);
    let (r, p) = t.alloc(PAGE * 2, end, PAGE, u64::MAX - (PAGE - 1), MTYPE_C);
    assert!(r < 0 && p == u64::MAX, "out-of-range alignment fails without output");
}

#[test]
fn direct_memory_type_of_an_offset() {
    let t = T::new();
    let a = t.alloc_ok(PAGE, PAGE);
    let b = t.alloc_ok(PAGE, PAGE);
    let (ty, start, end) = (t.base + 0x200, t.base + 0x208, t.base + 0x210);
    let (r, ctx) = t.call(cb_get_direct_memory_type, &[b + 0x10, ty, start, end]);
    assert_eq!(r, OK);
    assert_eq!((ctx.read_u32(ty), ctx.read_u64(start), ctx.read_u64(end)), (Some(MTYPE_C as u32), Some(b), Some(b + PAGE)));
    let (r, ctx) = t.call(cb_get_direct_memory_type, &[a, ty, start, end]);
    assert_eq!(r, OK);
    assert_eq!((ctx.read_u64(start), ctx.read_u64(end)), (Some(a), Some(a + 2 * PAGE)), "same-type blocks after it merge");
    assert_eq!(t.call(cb_get_direct_memory_type, &[a, 0, start, end]).0, EINVAL);
    assert_eq!(t.release(a, 2 * PAGE), OK);
    assert_eq!(t.call(cb_get_direct_memory_type, &[a, ty, start, end]).0, ENOENT);
    assert_eq!(t.call(cb_get_direct_memory_type, &[u64::MAX, ty, start, end]).0, ENOENT);
}

#[test]
fn direct_memory_query_merges_same_type_blocks() {
    let t = T::new();
    let a = t.alloc_ok(PAGE, PAGE);
    let b = t.alloc_ok(PAGE, PAGE);
    assert_eq!(b, a + PAGE);
    let info = t.base + 0x200;
    let (r, ctx) = t.call(cb_direct_memory_query, &[a, 0, info, 24]);
    assert_eq!(r, OK);
    assert_eq!((ctx.read_u64(info), ctx.read_u64(info + 8), ctx.read_u32(info + 16)), (Some(a), Some(a + 2 * PAGE), Some(MTYPE_C as u32)));
    assert_eq!(t.call(cb_direct_memory_query, &[a, 0, info, 16]).0, EINVAL);
    assert_eq!(t.call(cb_direct_memory_query, &[a, 2, info, 24]).0, EINVAL);
    assert_eq!(t.release(a, 2 * PAGE), OK);
    assert_eq!(t.call(cb_direct_memory_query, &[a, 0, info, 24]).0, EACCES);
    // Find-next beyond every allocation reports the terminal entry.
    let (r, ctx) = t.call(cb_direct_memory_query, &[direct_size() - PAGE, 1, info, 24]);
    assert_eq!(r, OK);
    assert_eq!((ctx.read_u64(info), ctx.read_u64(info + 8)), (Some(direct_size()), Some(direct_size())));
}

#[test]
fn hintless_direct_map_uses_canonical_base_and_content_persists() {
    let t = T::new();
    let (r, phys) = t.alloc(0, 0x2_6000_0000, 0x40_0000, 0x20_0000, 12);
    assert_eq!(r, OK);
    let (r, base) = t.direct(0, 0x40_0000, PROT_CPU_RW, 0, phys, 0x20_0000, "libc_heap");
    assert_eq!(r, OK);
    assert!(base >= GUEST_DEFAULT_MAP_BASE, "hint-less map below the canonical base: {base:#x}");
    assert_eq!(base % 0x20_0000, 0);
    for off in (0..PAGE * 4).step_by(8) {
        t.write(base + off, off ^ 0x4b59_5459_5045_525a);
    }
    assert_eq!(t.munmap(base, 0x40_0000), OK);
    let (r, again) = t.direct(0, PAGE * 4, PROT_CPU_RW, 0, phys, PAGE, "persist_b");
    assert_eq!(r, OK);
    for off in (0..PAGE * 4).step_by(8) {
        assert_eq!(t.read(again + off), off ^ 0x4b59_5459_5045_525a, "content survives remap");
    }
    assert_eq!(t.munmap(again, PAGE * 4), OK);
    let (r, part) = t.direct(0, PAGE, PROT_CPU_RW, 0, phys + 2 * PAGE, PAGE, "persist_c");
    assert_eq!(r, OK);
    assert_eq!(t.read(part), (2 * PAGE) ^ 0x4b59_5459_5045_525a);
    assert_eq!(t.munmap(part, PAGE), OK);
    // Repeated map/unmap returns the same address.
    let mut first = 0;
    for i in 0..16 {
        let (r, a) = t.direct(0, PAGE, PROT_CPU_RW, 0, phys, PAGE, "reuse_direct");
        assert_eq!(r, OK);
        if i == 0 {
            first = a;
        }
        assert_eq!(a, first);
        assert_eq!(t.munmap(a, PAGE), OK);
    }
    assert_eq!(t.release(phys, 0x40_0000), OK);
}

#[test]
fn fixed_direct_replacement_in_place() {
    let t = T::new();
    let phys = t.alloc_ok(PAGE * 4, PAGE);
    let (_, alias) = t.direct(0, PAGE * 4, PROT_CPU_RW, 0, phys, PAGE, "replacement_alias");
    let (_, base) = t.direct(0, PAGE * 3, PROT_CPU_RW, 0, phys, PAGE, "replacement_original");
    for i in 0..4 {
        t.write(alias + i * PAGE, (i + 1) * 0x11);
    }
    assert_eq!(t.direct(base + PAGE, PAGE, PROT_CPU_RD, MAP_FIXED, phys + PAGE, PAGE, "replacement_same").0, OK);
    let q = t.query(base + PAGE, 0);
    assert_eq!((q.start, q.end, q.prot, q.offset, q.name.as_str()), (base + PAGE, base + 2 * PAGE, PROT_CPU_RD, phys + PAGE, "replacement_same"));
    assert_eq!(t.direct(base + PAGE, PAGE, PROT_CPU_RW, MAP_FIXED, phys + 3 * PAGE, PAGE, "replacement_changed").0, OK);
    assert_eq!(t.read(base + PAGE), 0x44, "the new physical offset is visible");
    t.write(base + PAGE, 0x45);
    assert_eq!(t.read(alias + 3 * PAGE), 0x45);
    assert_eq!(t.read(alias + PAGE), 0x22);
    assert_eq!(t.read(base), 0x11);
    assert_eq!(t.direct(base + PAGE, PAGE * 2, PROT_CPU_RD, MAP_FIXED, phys + PAGE, PAGE, "replacement_span").0, OK);
    assert_eq!((t.read(base), t.read(base + PAGE), t.read(base + 2 * PAGE)), (0x11, 0x22, 0x33));
    let q = t.query(base + PAGE, 0);
    assert_eq!((q.start, q.end, q.offset, q.name.as_str()), (base + PAGE, base + 3 * PAGE, phys + PAGE, "replacement_span"));
    assert_eq!(t.munmap(base, PAGE * 3), OK);
    assert_eq!(t.munmap(alias, PAGE * 4), OK);
    assert_eq!(t.release(phys, PAGE * 4), OK);
}

#[test]
fn fixed_reserve_replaces_part_of_a_direct_mapping() {
    let t = T::new();
    let (total, keep) = (PAGE * 13, PAGE * 5);
    let replace = total - keep;
    let phys = t.alloc_ok(total, PAGE);
    let (_, alias) = t.direct(0, total, PROT_CPU_RW, 0, phys, PAGE, "partial_replace_alias");
    t.write(alias + keep, 0x4b59_5459_414c_4941);
    let (_, base) = t.reserve(0, total, 0, PAGE);
    let (r, mapped) = t.direct(base, total, PROT_CPU_RW, MAP_FIXED | MAP_NO_COALESCE, phys, PAGE, "partial_replace_direct");
    assert_eq!((r, mapped), (OK, base));
    t.write(base, 0x4b59_5459_4b45_4550);
    let (r, rep) = t.reserve(base + keep, replace, MAP_FIXED | MAP_NO_COALESCE, PAGE);
    assert_eq!((r, rep), (OK, base + keep));
    assert_eq!(t.read(base), 0x4b59_5459_4b45_4550, "the neighbour survives");
    assert_eq!(t.query(base, 0).shape(), (base, base + keep, reported_protection(PROT_CPU_RW), false, true, false, true));
    assert_eq!(t.query(base + keep, 0).shape(), (base + keep, base + total, 0, false, false, false, false));
    let (r, remapped) = t.direct(rep, replace, PROT_CPU_RW, MAP_FIXED | MAP_NO_COALESCE, phys + keep, PAGE, "partial_replace_remap");
    assert_eq!((r, remapped), (OK, rep));
    assert_eq!(t.read(remapped), 0x4b59_5459_414c_4941, "backing offset preserved");
    t.write(remapped, 0x4b59_5459_5245_5553);
    assert_eq!(t.read(alias + keep), 0x4b59_5459_5245_5553);
    assert_eq!(t.munmap(base, keep), OK);
    assert_eq!(t.munmap(base + keep, replace), OK);
    assert_eq!(t.munmap(alias, total), OK);
    assert_eq!(t.release(phys, total), OK);
}

#[test]
fn extended_and_user_mappings_do_not_alias() {
    let t = T::new();
    let addresses = [0x70_0000_0000u64, EXTENDED_MEMORY_BASE + 0x70_0000_0000, EXTENDED_MEMORY_BASE + EXTENDED_MEMORY_SIZE - PAGE];
    let phys = t.alloc_ok(PAGE * 3, PAGE);
    for (i, &a) in addresses.iter().enumerate() {
        let (r, m) = t.direct(a, PAGE, PROT_CPU_RW, MAP_FIXED | MAP_NO_OVERWRITE, phys + i as u64 * PAGE, PAGE, "ext");
        assert_eq!((r, m), (OK, a));
        t.write(a, 0x0123_4567_0000_0000 + i as u64);
    }
    for (i, &a) in addresses.iter().enumerate() {
        assert_eq!(t.read(a), 0x0123_4567_0000_0000 + i as u64);
        assert_eq!(t.munmap(a, PAGE), OK);
    }
    assert_eq!(t.release(phys, PAGE * 3), OK);
}

#[test]
fn large_hinted_reserve_hosts_small_direct_map() {
    let t = T::new();
    let (arena_base, arena_size, window_size) = (HOST_USER_MIN, 0x400_0000u64, 0x20_0000u64);
    let (r, arena) = t.reserve(arena_base, arena_size, 0, 0x20_0000);
    assert_eq!(r, OK);
    assert!(arena >= arena_base && arena % 0x20_0000 == 0);
    let (r, window) = t.reserve(arena_base, window_size, 0, PAGE);
    assert_eq!(r, OK);
    assert!(window >= arena + arena_size, "second hinted reserve skips the first");
    let phys = t.alloc_ok(PAGE * 2, PAGE);
    let (r, m) = t.direct(window, PAGE * 2, PROT_CPU_RW, MAP_FIXED | MAP_NO_COALESCE, phys, 0, "large_reserve");
    assert_eq!((r, m), (OK, window));
    assert_eq!(t.munmap(window, PAGE * 2), OK);
    assert_eq!(t.release(phys, PAGE * 2), OK);
    assert_eq!(t.munmap(window + PAGE * 2, window_size - PAGE * 2), OK);
    assert_eq!(t.munmap(arena, arena_size), OK);
}

#[test]
fn batch_map_operations_and_partial_failure() {
    let t = T::new();
    let phys = t.alloc_ok(PAGE, PAGE);
    let (_, base) = t.reserve(0, PAGE * 2, 0, PAGE);
    let entries = t.base + 0x1000;
    let put = |i: u64, start: u64, offset: u64, len: u64, prot: u8, mtype: u8, op: i32| {
        let e = entries + i * 32;
        let ctx = t.ctx();
        ctx.write_u64(e, start);
        ctx.write_u64(e + 8, offset);
        ctx.write_u64(e + 16, len);
        ctx.write_bytes(e + 24, &[prot, mtype, 0, 0]);
        ctx.write_u32(e + 28, op as u32);
    };
    put(0, base, phys, PAGE, PROT_CPU_RW as u8, 0, 0);
    put(1, base + PAGE, 0, PAGE, PROT_CPU_RW as u8, 0, 3);
    put(2, base, 0, PAGE, PROT_CPU_RD as u8, 0, 2);
    put(3, base + PAGE, 0, PAGE, PROT_CPU_RD as u8, 2, 4);
    let processed = t.base + 0x50;
    let (r, ctx) = t.call(cb_batch_map, &[entries, 4, processed]);
    assert_eq!(r, OK);
    assert_eq!(ctx.read_u32(processed), Some(4));
    let q = t.query(base, 0);
    assert_eq!(q.shape(), (base, base + PAGE, PROT_CPU_RD, false, true, false, true));
    assert_eq!((q.name.as_str(), q.offset), ("anon", phys));
    assert_eq!(t.query(base + PAGE, 0).shape(), (base + PAGE, base + 2 * PAGE, PROT_CPU_RD, true, false, false, true));
    put(0, base, 0, PAGE, 0, 0, 1);
    put(1, 0, phys, PAGE, PROT_CPU_RW as u8, 0, 0);
    put(2, base + PAGE, 0, PAGE, 0, 0, 1);
    let (r, ctx) = t.call(cb_batch_map, &[entries, 3, processed]);
    assert_eq!(r, EINVAL);
    assert_eq!(ctx.read_u32(processed), Some(1), "completed prefix counted");
    t.expect_unmapped(base);
    assert!(t.query(base + PAGE, 0).flexible, "nothing after the failure ran");
    let (r, ctx) = t.call(cb_batch_map2, &[entries + 64, 1, processed, 0]);
    assert_eq!(r, OK);
    assert_eq!(ctx.read_u32(processed), Some(1));
    t.expect_unmapped(base + PAGE);
    assert_eq!(t.release(phys, PAGE), OK);
}

#[test]
fn memory_pool_contracts_and_commit_cycle() {
    let t = T::new();
    let out = t.base + 0x60;
    let expand = |len: u64, align: u64| {
        t.ctx().write_u64(out, u64::MAX);
        let (r, ctx) = t.call(cb_memory_pool_expand, &[0, direct_size(), len, align, out]);
        (r, ctx.read_u64(out).unwrap())
    };
    let reserve = |addr: u64, len: u64| {
        let (r, ctx) = t.call(cb_memory_pool_reserve, &[addr, len, 0, 0, out]);
        (r, ctx.read_u64(out).unwrap())
    };
    let commit = |addr: u64, len: u64, prot: i32| t.call(cb_memory_pool_commit, &[addr, len, MTYPE_C as u64, prot as u64, 0]).0;
    let decommit = |addr: u64, len: u64| t.call(cb_memory_pool_decommit, &[addr, len, 0]).0;
    let stats = || {
        let s = t.base + 0x70;
        let (r, ctx) = t.call(cb_memory_pool_get_block_stats, &[s, 16]);
        assert_eq!(r, OK);
        (ctx.read_u32(s).unwrap() as i32, ctx.read_u32(s + 8).unwrap() as i32)
    };
    assert_eq!(reserve(0, PAGE).0, EINVAL, "reserve length is a 2 MiB multiple");
    let (r, base) = reserve(0, POOL_RESERVE_LEN);
    assert_eq!(r, OK);
    assert_eq!(commit(base, PAGE, PROT_CPU_RW), EINVAL, "commit length is a 64 KiB multiple");
    assert_eq!(decommit(base, PAGE), EINVAL);
    assert_eq!(t.munmap(base, POOL_RESERVE_LEN), OK);

    assert_eq!(expand(PAGE, POOL_ALIGN).0, EINVAL);
    assert_eq!(expand(POOL_EXPAND_LEN, PAGE).0, EINVAL);
    assert_eq!(expand(POOL_EXPAND_LEN, POOL_ALIGN * 3).0, EINVAL);
    let (r, pool) = expand(POOL_EXPAND_LEN, POOL_ALIGN);
    assert_eq!(r, OK);
    assert_eq!(pool % POOL_ALIGN, 0);
    let io = t.base + 0x28;
    t.ctx().write_u64(io, 0);
    assert!(t.call(cb_map_direct_memory, &[io, POOL_COMMIT_LEN, PROT_CPU_RW as u64, 0, pool, POOL_ALIGN]).0 < 0, "pool pages are not direct memory");
    assert_eq!(stats(), ((POOL_EXPAND_LEN / POOL_ALIGN) as i32, 0));

    let (r, base) = reserve(HOST_USER_MIN, POOL_RESERVE_LEN);
    assert_eq!(r, OK);
    let q = t.query(base, 0);
    assert!(q.pooled && !q.committed, "pool reserve is pooled and uncommitted");
    let flex = t.available_flexible();
    let len = POOL_COMMIT_LEN * 2;
    assert_eq!(commit(base, len, PROT_CPU_RW), OK);
    assert_eq!(t.query(base, 0).shape(), (base, base + len, reported_protection(PROT_CPU_RW), false, false, true, true));
    assert_eq!(t.available_flexible(), flex, "pool commits use expanded direct memory");
    assert_eq!(t.checked_release(pool, POOL_EXPAND_LEN), ENOENT, "committed expansion cannot be released");
    t.write(base, 0x504f_4f4c_4241_434b);
    t.write(base + POOL_COMMIT_LEN, 0x5348_4152_4544_4d45);
    assert_eq!(decommit(base, POOL_COMMIT_LEN), OK);
    assert_eq!(t.query(base, 0).shape(), (base, base + POOL_COMMIT_LEN, 0, false, false, true, false));
    assert_eq!(t.query(base + POOL_COMMIT_LEN, 0).shape(), (base + POOL_COMMIT_LEN, base + len, reported_protection(PROT_CPU_RW), false, false, true, true));
    assert_eq!(stats(), ((POOL_EXPAND_LEN / POOL_ALIGN) as i32 - 1, 1));
    assert_eq!(commit(base, POOL_COMMIT_LEN, PROT_CPU_RW), OK);
    assert_eq!(t.read(base), 0x504f_4f4c_4241_434b, "recommitted pool pages keep their contents");
    assert_eq!(t.read(base + POOL_COMMIT_LEN), 0x5348_4152_4544_4d45);
    assert_eq!(decommit(base, len), OK);
    assert_eq!(t.munmap(base, POOL_RESERVE_LEN), OK);
    assert_eq!(t.release(pool, POOL_EXPAND_LEN), OK);
    assert_eq!(stats(), (0, 0), "released expansion leaves the pool budget");
}

/// sceKernelMemoryPoolBatch: a commit entry maps with its memory type, a
/// type-protect entry changes type and protection together.
#[test]
fn memory_pool_batch_forwards_memory_types() {
    let t = T::new();
    let out = t.base + 0x60;
    let (r, ctx) = t.call(cb_memory_pool_expand, &[0, direct_size(), POOL_COMMIT_LEN, POOL_ALIGN, out]);
    assert_eq!(r, OK);
    let pool = ctx.read_u64(out).unwrap();
    let (r, ctx) = t.call(cb_memory_pool_reserve, &[0, POOL_RESERVE_LEN, 0, 0, out]);
    assert_eq!(r, OK);
    let base = ctx.read_u64(out).unwrap();
    let (entries, done) = (t.base + 0x400, t.base + 0x70);
    let entry = |ctx: &CpuContext, i: u64, op: u32, prot: u8, mtype: u8| {
        let e = entries + i * 32;
        ctx.write_bytes(e, &[0u8; 32]);
        ctx.write_u32(e, op);
        ctx.write_u64(e + 8, base);
        ctx.write_u64(e + 16, POOL_COMMIT_LEN);
        ctx.write_bytes(e + 24, &[prot, mtype]);
    };
    entry(&ctx, 0, 1, PROT_CPU_RW as u8, 3);
    entry(&ctx, 1, 4, PROT_CPU_RD as u8, 6);
    let (r, ctx) = t.call(cb_memory_pool_batch, &[entries, 1, done, 0]);
    assert_eq!(r, OK);
    assert_eq!(ctx.read_u32(done), Some(1));
    let q = t.query(base, 0);
    assert_eq!((q.prot, q.mtype), (reported_protection(PROT_CPU_RW), 3), "the commit entry's type");
    let (r, ctx) = t.call(cb_memory_pool_batch, &[entries + 32, 1, done, 0]);
    assert_eq!((r, ctx.read_u32(done)), (OK, Some(1)));
    let q = t.query(base, 0);
    assert_eq!((q.prot, q.mtype), (PROT_CPU_RD, 6), "type and protection of the type-protect entry");
    assert_eq!(t.call(cb_memory_pool_decommit, &[base, POOL_COMMIT_LEN, 0]).0, OK);
    assert_eq!(t.munmap(base, POOL_RESERVE_LEN), OK);
    assert_eq!(t.release(pool, POOL_COMMIT_LEN), OK);
}

#[test]
fn memory_pool_fragmented_backing_and_multi_range_decommit() {
    let t = T::new();
    let out = t.base + 0x60;
    let expand = |len: u64| {
        let (r, ctx) = t.call(cb_memory_pool_expand, &[0, direct_size(), len, POOL_ALIGN, out]);
        assert_eq!(r, OK);
        ctx.read_u64(out).unwrap()
    };
    let first = expand(POOL_COMMIT_LEN);
    let (_, gap) = t.alloc(0, direct_size(), POOL_COMMIT_LEN, POOL_ALIGN, MTYPE_C);
    let second = expand(POOL_COMMIT_LEN);
    assert!(first + POOL_COMMIT_LEN == gap && gap + POOL_COMMIT_LEN == second);
    let (r, ctx) = t.call(cb_memory_pool_reserve, &[0, POOL_RESERVE_LEN, 0, 0, out]);
    assert_eq!(r, OK);
    let base = ctx.read_u64(out).unwrap();
    let commit = |addr: u64, len: u64, prot: i32| t.call(cb_memory_pool_commit, &[addr, len, MTYPE_C as u64, prot as u64, 0]).0;
    let decommit = |addr: u64, len: u64| t.call(cb_memory_pool_decommit, &[addr, len, 0]).0;
    let len = POOL_COMMIT_LEN * 2;
    assert_eq!(commit(base, len, PROT_CPU_RW), OK);
    assert_eq!(commit(base + len, POOL_COMMIT_LEN, PROT_CPU_RW), ENOMEM, "pool exhausted");
    assert_eq!(t.query(base + len, 0).shape(), (base + len, base + POOL_RESERVE_LEN, 0, false, false, true, false));
    assert_eq!(decommit(base, len), OK);
    // Mixed protections and a reserved/committed mix decommit together.
    assert_eq!(commit(base, POOL_COMMIT_LEN, PROT_CPU_RW), OK);
    assert_eq!(commit(base + POOL_COMMIT_LEN, POOL_COMMIT_LEN, PROT_CPU_RD), OK);
    assert_eq!(decommit(base, len), OK);
    let q = t.query(base, 0);
    assert!(q.start <= base && q.end >= base + len && q.pooled && !q.committed);
    assert_eq!(commit(base + POOL_COMMIT_LEN, POOL_COMMIT_LEN, PROT_CPU_RD), OK);
    assert_eq!(decommit(base, len), OK);
    // A non-pool range inside the span fails the whole decommit.
    assert_eq!(commit(base, POOL_COMMIT_LEN, PROT_CPU_RW), OK);
    assert_eq!(t.munmap(base + POOL_COMMIT_LEN, POOL_COMMIT_LEN), OK);
    assert_eq!(t.flex(base + POOL_COMMIT_LEN, POOL_COMMIT_LEN, PROT_CPU_RD, MAP_FIXED, Some("pool_invalid_tail")).0, OK);
    assert_eq!(decommit(base, len), EACCES);
    assert_eq!(t.query(base, 0).shape(), (base, base + POOL_COMMIT_LEN, reported_protection(PROT_CPU_RW), false, false, true, true));
    assert_eq!(decommit(base, POOL_COMMIT_LEN), OK);
    assert_eq!(t.munmap(base, POOL_RESERVE_LEN), OK);
    for p in [first, second, gap] {
        assert_eq!(t.release(p, POOL_COMMIT_LEN), OK);
    }
}

#[test]
fn automatic_memory_reuse_and_aliases() {
    let t = T::new();
    let block = 0x20_0000u64;
    let base = EXTENDED_MEMORY_BASE + 0x2000_0000;
    let physical = allocate_direct_memory(0, direct_size() as i64, block, block, 0, true).unwrap();
    assert_eq!(map_automatic_memory(base, block, MTYPE_C, PROT_CPU_RW), OK);
    t.write(base + PAGE * 3, 0x0123_4567_89ab_cdef);
    let (r, first_alias) = t.direct(base + block * 2, PAGE * 3, PROT_CPU_RW, MAP_FIXED, physical + PAGE, PAGE, "a");
    assert_eq!(r, OK);
    let (r, second_alias) = t.direct(base + block * 3, PAGE * 2, PROT_CPU_RW, MAP_FIXED, physical + PAGE * 2, PAGE, "b");
    assert_eq!(r, OK);
    assert_eq!(map_automatic_memory(base + block * 4, PAGE, MTYPE_C, PROT_CPU_RW), EAGAIN, "mapped pages are not automatic");
    assert_eq!(t.munmap(base, block), OK);
    assert_eq!(map_automatic_memory(base, block, MTYPE_C, PROT_CPU_RW), EAGAIN, "aliased pages stay unavailable");
    assert_eq!(map_automatic_memory(base, block - PAGE * 3, MTYPE_C, PROT_CPU_RW), OK);
    assert_eq!((t.query(base, 0).offset, t.query(base + PAGE, 0).offset), (physical, physical + PAGE * 4));
    assert_eq!(t.munmap(first_alias, PAGE * 3), OK);
    assert_eq!(map_automatic_memory(base + block * 4, PAGE, MTYPE_C, PROT_CPU_RW), OK);
    assert_eq!(t.query(base + block * 4, 0).offset, physical + PAGE);
    assert_eq!(t.munmap(second_alias, PAGE), OK);
    assert_eq!(map_automatic_memory(base + block * 5, PAGE, MTYPE_C, PROT_CPU_RW), OK);
    assert_eq!(t.query(base + block * 5, 0).offset, physical + PAGE * 2);
    assert_eq!(t.read(base + block * 3 + PAGE), 0x0123_4567_89ab_cdef, "surviving alias unchanged");
    assert_eq!(t.checked_release(physical, block), OK);
    t.expect_unmapped(base);
    t.expect_unmapped(base + block * 3 + PAGE);
    assert_eq!(map_automatic_memory(base, PAGE, MTYPE_C, PROT_CPU_RW), EAGAIN);
    let (r, reused) = t.alloc(physical, physical + block, block, block, MTYPE_C);
    assert_eq!((r, reused), (OK, physical));
    assert_eq!(t.checked_release(reused, block), OK);
}

/// sceKernelGetPageTableStats counts entries as the console does: 8192 of
/// each; CPU one per 2 MiB granule; GPU one per 1 GiB region plus one per
/// granule that is not a whole 2 MiB-aligned direct page.
#[test]
fn page_table_stats_count_granules_and_large_pages() {
    let t = T::new();
    let s = t.base + 0x80;
    let used = || {
        let (r, ctx) = t.call(cb_get_page_table_stats, &[s, s + 4, s + 8, s + 12]);
        assert_eq!(r, OK);
        assert_eq!((ctx.read_u32(s), ctx.read_u32(s + 8)), (Some(8192), Some(8192)));
        (8192 - ctx.read_u32(s + 4).unwrap() as i64, 8192 - ctx.read_u32(s + 12).unwrap() as i64)
    };
    const GIB: u64 = 1 << 30;
    const MIB2: u64 = 2 << 20;
    let (r, area) = t.reserve(0, 4 * GIB, 0, GIB);
    assert_eq!(r, OK);
    let (c0, g0) = used();
    let phys = t.alloc_ok(8 * MIB2, MIB2);
    // 4 MiB, 2 MiB-aligned VA and physical memory, CPU and GPU access.
    assert_eq!(t.direct(area, 2 * MIB2, 0x33, MAP_FIXED, phys, 0, "pts_a").0, OK);
    assert_eq!(used(), (c0 + 2, g0 + 1));
    assert_eq!(t.munmap(area, 2 * MIB2), OK);
    // The same with physical memory 64 KiB off a 2 MiB boundary.
    let area2 = area + GIB;
    assert_eq!(t.direct(area2, 2 * MIB2, 0x33, MAP_FIXED, phys + 0x10000, 0, "pts_b").0, OK);
    assert_eq!(used(), (c0 + 2, g0 + 3));
    assert_eq!(t.munmap(area2, 2 * MIB2), OK);
    // CPU-only flexible memory: no GPU entries; two maps in one granule
    // count once.
    let area3 = area + 2 * GIB;
    assert_eq!(t.flex(area3, PAGE, PROT_CPU_RW, MAP_FIXED, Some("pts_c")).0, OK);
    assert_eq!(used(), (c0 + 1, g0));
    assert_eq!(t.flex(area3 + 4 * PAGE, PAGE, PROT_CPU_RW, MAP_FIXED, Some("pts_d")).0, OK);
    assert_eq!(used(), (c0 + 1, g0));
    assert_eq!(t.munmap(area3, 5 * PAGE), OK);
    assert_eq!(t.munmap(area, 4 * GIB), OK);
    assert_eq!(t.release(phys, 8 * MIB2), OK);
    assert_eq!(used(), (c0, g0));
}

#[test]
fn virtual_range_names_page_table_stats_and_prt() {
    let t = T::new();
    let base = t.flex_ok(PAGE * 2, PROT_CPU_RW | PROT_GPU_READ, "");
    let n = t.name(2, "renamed");
    assert_eq!(t.call(cb_set_virtual_range_name, &[base, PAGE, n]).0, OK);
    assert_eq!(t.query(base, 0).name, "renamed");
    assert_eq!(t.query(base + PAGE, 0).name, "");
    assert_eq!(t.call(cb_set_virtual_range_name, &[base, PAGE, 0]).0, EFAULT);
    let long = t.name(3, &"y".repeat(32));
    assert_eq!(t.call(cb_set_virtual_range_name, &[base, PAGE, long]).0, ENAMETOOLONG);
    assert_eq!(t.call(cb_clear_virtual_range_name, &[base, PAGE]).0, OK);
    assert_eq!(t.query(base, 0).end, base + 2 * PAGE, "cleared name merges the range again");
    let s = t.base + 0x80;
    let (r, ctx) = t.call(cb_get_page_table_stats, &[s, s + 4, s + 8, s + 12]);
    assert_eq!(r, OK);
    let total = ctx.read_u32(s).unwrap() as i32;
    assert_eq!(total, PAGE_TABLE_POOL_ENTRIES);
    assert!((ctx.read_u32(s + 12).unwrap() as i32) < total, "GPU-visible mapping uses GPU entries");
    assert_eq!(t.call(cb_get_page_table_stats, &[0, s + 4, s + 8, s + 12]).0, EFAULT);
    // Query/protection report the end exclusively.
    let (r, ctx) = t.call(cb_query_memory_protection, &[base + 8, s, s + 8, s + 16]);
    assert_eq!(r, OK);
    assert_eq!((ctx.read_u64(s), ctx.read_u64(s + 8)), (Some(base), Some(base + 2 * PAGE)));
    assert_eq!(t.munmap(base, PAGE * 2), OK);
    // PRT apertures.
    assert_eq!(t.call(cb_set_prt_aperture, &[3, PRT_APERTURE_START, PAGE]).0, EINVAL);
    assert_eq!(t.call(cb_set_prt_aperture, &[0, PRT_APERTURE_START - PAGE, PAGE]).0, EINVAL);
    assert_eq!(t.call(cb_set_prt_aperture, &[1, PRT_APERTURE_START, PAGE * 4]).0, OK);
    let (r, ctx) = t.call(cb_get_prt_aperture, &[1, s, s + 8]);
    assert_eq!(r, OK);
    assert_eq!((ctx.read_u64(s), ctx.read_u64(s + 8)), (Some(PRT_APERTURE_START), Some(PAGE * 4)));
    assert_eq!(t.call(cb_get_prt_aperture, &[1, 0, s + 8]).0, EFAULT);
    assert_eq!(t.call(cb_set_prt_aperture, &[1, 0, 0]).0, OK);
}

#[test]
fn stacks_are_reported() {
    let t = T::new();
    let base = 0x5CC8_0000_0000u64;
    assert!(register_stack(base, PAGE * 4));
    let q = t.query(base + PAGE, 0);
    assert!(q.stack && q.committed && !q.flexible);
    let (s, e) = (t.base + 0x60, t.base + 0x68);
    let (r, ctx) = t.call(cb_is_stack, &[base + 8, s, e]);
    assert_eq!(r, OK);
    assert_eq!((ctx.read_u64(s), ctx.read_u64(e)), (Some(base), Some(base + PAGE * 4)));
    unregister_stack(base, PAGE * 4);
    t.expect_unmapped(base);
}

#[test]
fn agc_init_installs_the_driver_system_table() {
    use crate::agc::defaults::{release_driver_memory, reserve_driver_memory, system_table, AGC_DMEM_OFFSET, DRIVER_DMEM_BASE, DRIVER_DMEM_SIZE, QUEUE_VALIDATOR};
    let t = T::new();
    reserve_driver_memory();
    // Shared driver backing: named direct memory of type 12, CPU+GPU RW.
    let q = t.query(DRIVER_DMEM_BASE, 0);
    assert_eq!((q.start, q.end, q.mtype, q.prot, q.direct, q.name.as_str()), (DRIVER_DMEM_BASE, DRIVER_DMEM_BASE + DRIVER_DMEM_SIZE, 12, 0x33, true, "SceAgcDriver"));
    let table = DRIVER_DMEM_BASE + AGC_DMEM_OFFSET;
    let word = |a: u64| (t.read(a & !7) >> ((a & 4) * 8)) as u32;
    // The descriptor reaches the function entries (16-byte stride, 8 records).
    let descriptor = [word(table), word(table + 4), word(table + 8), word(table + 12)];
    let functions = descriptor[0] as u64 | ((descriptor[1] & 0xFFFF) as u64) << 32;
    assert_eq!(functions, table + 0x60);
    assert_eq!((descriptor[1] >> 16) & 0x3FFF, 16, "stride");
    assert_eq!((descriptor[2], descriptor[3]), (8, 0x5204));
    assert_eq!(descriptor, system_table(table).0);
    // Entry 2: the queue validator, 256-byte aligned, no user data.
    let code = t.read(functions + 2 * 16);
    assert!(code != 0 && code & 0xFF == 0);
    assert_eq!(t.read(functions + 2 * 16 + 8), 0);
    let words: Vec<u32> = (0..16).map(|i| word(code + i * 4)).collect();
    assert_eq!(words, QUEUE_VALIDATOR);
    // Unused vertex validation is off at startup, and a second start keeps
    // the existing system state.
    let instrumentation = table + 0x18;
    assert_eq!(t.read(instrumentation), 0);
    t.write(instrumentation, 0x1234_0000);
    reserve_driver_memory();
    assert_eq!(t.read(instrumentation), 0x1234_0000, "repeat initialization reset existing system state");
    release_driver_memory();
    t.expect_unmapped(DRIVER_DMEM_BASE);
}

/// The static buffers are the 16-byte slots of the system table (slot 0:
/// its descriptor, stride 16), the static qwords start at slot 1; index 6
/// and up is refused for buffers (reference libSceAgc tests/StaticBuffer.cpp).
#[test]
fn agc_static_buffers_and_qwords_live_in_the_system_table() {
    use crate::agc::defaults::{get_static_buffer, get_static_qword, release_driver_memory, reserve_driver_memory, set_static_buffer, set_static_qword, AGC_DMEM_OFFSET, DRIVER_DMEM_BASE};
    const INVALID: HleResult = 0x8A6C_000Bu32 as i32 as i64;
    let t = T::new();
    reserve_driver_memory();
    let table = DRIVER_DMEM_BASE + AGC_DMEM_OFFSET;
    let slot = |i: u64| [t.read(table + i * 16), t.read(table + i * 16 + 8)];
    let buf = t.base + 0x300;
    assert_eq!(t.call(get_static_buffer, &[0, buf]).0, OK);
    assert_eq!([t.read(buf), t.read(buf + 8)], slot(0));
    assert_eq!(t.read(buf) >> 48, 16, "static buffer 0 is the system table descriptor");
    let fetch_table = slot(0);
    t.write(buf, 0x1122_3344_5566_7788);
    t.write(buf + 8, 0x99aa_bbcc_ddee_ff00);
    assert_eq!(t.call(set_static_buffer, &[5, buf]).0, OK);
    assert_eq!(slot(5), [0x1122_3344_5566_7788, 0x99aa_bbcc_ddee_ff00]);
    let back = t.base + 0x320;
    assert_eq!(t.call(get_static_buffer, &[5, back]).0, OK);
    assert_eq!([t.read(back), t.read(back + 8)], slot(5));
    assert_eq!(t.call(set_static_buffer, &[6, buf]).0, INVALID);
    assert_eq!(t.call(get_static_buffer, &[6, back]).0, INVALID);
    assert_eq!(slot(0), fetch_table, "buffer writes changed slot 0");
    assert_eq!(t.call(set_static_qword, &[0, 0x0123_4567_89ab_cdef]).0, OK);
    assert_eq!(t.call(set_static_qword, &[3, 0xfedc_ba98_7654_3210]).0, OK);
    assert_eq!((slot(1)[0], slot(2)[1]), (0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210));
    let q = t.base + 0x340;
    assert_eq!(t.call(get_static_qword, &[3, q]).0, OK);
    assert_eq!(t.read(q), 0xfedc_ba98_7654_3210);
    assert_eq!(t.call(get_static_buffer, &[1, back]).0, OK);
    assert_eq!(t.read(back), 0x0123_4567_89ab_cdef, "qword 0 is the first half of buffer 1");
    release_driver_memory();
}

/// A failing call that a title retries is logged for its first 32 failures and
/// then at powers of two only.
#[test]
fn failed_memory_calls_are_logged_at_a_bounded_rate() {
    let logged: Vec<u64> = (0..1000).filter_map(|_| note_failure("rate-limit-test")).collect();
    let expected: Vec<u64> = (1..=32).chain([64, 128, 256, 512]).collect();
    assert_eq!(logged, expected);
}

/// Demon's Souls' memory tracking at start, as its trace shows it: three
/// exact windows that tile the whole direct memory. The last window is the
/// top `0x270000` bytes (`GetDirectMemorySize() - 0x270000`); the driver memory
/// must not be inside any of them, or the game retries its tracking setup
/// until its stack overflows.
#[test]
fn title_windows_tile_all_direct_memory_with_the_driver_reserved() {
    use crate::agc::defaults::{init as agc_init, release_driver_memory, reserve_driver_memory};
    const TRACKING: u64 = 0x27_0000;
    const ALLOCATOR_HIGH: u64 = 0x3_0000_0000;
    let t = T::new();
    reserve_driver_memory();
    let total = direct_size();
    assert_eq!(total, 0x3_2000_0000, "retail direct memory size");
    // Nothing of the title's direct memory is taken by the driver.
    assert_eq!(t.available_direct(0, total, 0x1_0000), (OK, 0, total), "the whole direct memory is available");
    // The driver block is system memory: the direct memory query does not show it.
    let info = t.base + 0x200;
    assert_eq!(t.call(cb_direct_memory_query, &[TOTAL_BYTES, 1, info, 24]).0, EACCES, "system memory is not direct memory");
    let (r, tracking) = t.alloc(total - TRACKING, total, TRACKING, 0x1_0000, MTYPE_C);
    assert_eq!((r, tracking), (OK, total - TRACKING), "tracking block at the top of direct memory");
    let low = total - TRACKING - ALLOCATOR_HIGH;
    let (r, low_block) = t.alloc(0, low, low, 0x1_0000, MTYPE_C);
    assert_eq!((r, low_block), (OK, 0), "allocator block from the bottom");
    let (r, high_block) = t.alloc(low, total - TRACKING, ALLOCATOR_HIGH, 0x1_0000, MTYPE_C);
    assert_eq!((r, high_block), (OK, low), "allocator block above it");
    let (r, _) = t.call(agc_init, &[0, 12]);
    assert_eq!(r, OK, "sceAgcInit after the title's windows");
    release_driver_memory();
    assert_eq!(t.release(tracking, TRACKING), OK);
    assert_eq!(t.release(low_block, low), OK);
    assert_eq!(t.release(high_block, ALLOCATOR_HIGH), OK);
}

/// A title that takes every byte of direct memory the system reports as
/// available (Demon's Souls' allocator does exactly that at start) still gets
/// a working sceAgcInit: the driver memory is reserved at process start,
/// before the title allocates anything, not out of what is left later.
#[test]
fn agc_driver_memory_survives_a_title_that_takes_all_direct_memory() {
    use crate::agc::defaults::{init as agc_init, release_driver_memory, reserve_driver_memory, DRIVER_DMEM_BASE};
    let t = T::new();
    // Process start (app_info::configure), before the title runs.
    reserve_driver_memory();
    let mut title = Vec::new();
    loop {
        let (r, phys, size) = t.available_direct(0, direct_size(), 0x1_0000);
        if r != OK {
            break;
        }
        let (r, p) = t.alloc(phys, phys + size, size, 0x1_0000, MTYPE_C);
        assert_eq!(r, OK, "title allocation of the {size:#x} bytes reported available");
        title.push((p, size));
    }
    assert!(!title.is_empty(), "the title allocated nothing");
    // sceAgcInit (version 12) comes after the title's allocations.
    let (r, _) = t.call(agc_init, &[0, 12]);
    assert_eq!(r, OK, "sceAgcInit after the title took all direct memory");
    let q = t.query(DRIVER_DMEM_BASE, 0);
    assert_eq!((q.start, q.name.as_str()), (DRIVER_DMEM_BASE, "SceAgcDriver"));
    release_driver_memory();
    for (p, s) in title {
        assert_eq!(t.release(p, s), OK, "release of the title's {s:#x} bytes");
    }
}

/// A fixed direct mapping that starts in a hole unmapped from an earlier
/// mapping of the same allocation replaces it; the pages around keep
/// their data (reference GuestMemory case).
#[test]
fn fixed_direct_map_over_a_hole_inside_a_mapping() {
    let t = T::new();
    let (r, phys) = t.alloc(0, direct_size(), PAGE * 4, 0, MTYPE_C);
    assert_eq!(r, OK);
    let (r, base) = t.direct(0, PAGE * 4, PROT_CPU_RW, 0, phys, 0, "holes");
    assert_eq!(r, OK);
    t.write(base, 0x11);
    t.write(base + PAGE * 3, 0x44);
    assert_eq!(t.munmap(base + PAGE, PAGE), OK);
    let (r, fixed) = t.direct(base + PAGE, PAGE * 2, PROT_CPU_RW, MAP_FIXED, phys + PAGE, 0, "holes");
    assert_eq!((r, fixed), (OK, base + PAGE));
    t.write(base + PAGE * 2, 0x33);
    assert_eq!((t.read(base), t.read(base + PAGE * 2), t.read(base + PAGE * 3)), (0x11, 0x33, 0x44));
    assert_eq!(t.munmap(base + PAGE, PAGE * 3), OK);
    assert_eq!(t.munmap(base, PAGE), OK);
    assert_eq!(t.release(phys, PAGE * 4), OK);
}

/// sceKernelAvailableDirectMemorySize reports the largest free block in the
/// range (aligned), not the first gap (reference GuestMemory case; no
/// block is ENOMEM here, as the console's kernel error for an empty range).
#[test]
fn available_direct_memory_is_the_largest_free_block() {
    let t = T::new();
    const WINDOW: u64 = 0x400_0000;
    let top = direct_size() - WINDOW;
    let (r, first) = t.alloc(top, top + WINDOW, 0x20_0000, 0x20_0000, MTYPE_C);
    assert_eq!((r, first), (OK, top));
    let (r, gap) = t.alloc(top, top + WINDOW, 0x40_0000, 0x20_0000, MTYPE_C);
    assert_eq!((r, gap), (OK, top + 0x20_0000));
    let (r, last) = t.alloc(top, top + WINDOW, 0x20_0000, 0x20_0000, MTYPE_C);
    assert_eq!((r, last), (OK, top + 0x60_0000));
    assert_eq!(t.release(gap, 0x40_0000), OK);
    assert_eq!(t.available_direct(top, top + WINDOW, 0), (OK, top + 0x80_0000, WINDOW - 0x80_0000));
    assert_eq!(t.available_direct(top, top + 0x80_0000, 0), (OK, top + 0x20_0000, 0x40_0000));
    assert_eq!(t.available_direct(top, top + WINDOW, 0x100_0000), (OK, top + 0x100_0000, WINDOW - 0x100_0000));
    assert_eq!(t.available_direct(top, top + 0x20_0000, 0), (ENOMEM, 0, 0));
    // A null physical-output pointer is allowed: only the size is written
    // back (measured on hardware).
    let (r, _) = t.call(cb_available_direct_memory_size, &[top, top + WINDOW, 0, 0, t.base + 0x98]);
    assert_eq!(r, OK);
    assert_eq!(t.release(first, 0x20_0000), OK);
    assert_eq!(t.release(last, 0x20_0000), OK);
    assert_eq!(t.available_direct(top, top + WINDOW, 0), (OK, top, WINDOW));
}

#[test]
fn export_nids_match_names() {
    for &(nid, name, _) in EXPORTS {
        assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
    }
}

const PROT_RW3: i32 = PROT_CPU_READ | PROT_CPU_WRITE;
const MAP_PRIVATE_ANON: u64 = 0x1002;

fn anon(len: u64) -> u64 {
    let a = posix_mmap(0, len, PROT_RW3, MAP_PRIVATE_ANON, -1, 0).expect("anonymous mmap");
    assert_ne!(a, 0);
    a
}

#[test]
fn non_fixed_hint_inside_a_reservation_maps_in_place() {
    let t = T::new();
    let (r, base) = t.reserve(0, PAGE * 4, 0, 0);
    assert_eq!(r, OK);
    let phys = t.alloc_ok(PAGE * 3, PAGE);
    let (r, direct) = t.direct(base + PAGE, PAGE, PROT_RW3, 0, phys, 0, "hint_direct");
    assert_eq!((r, direct), (OK, base + PAGE));
    t.write(direct, 21);
    let (r, flexible) = t.flex(base + PAGE * 2, PAGE, PROT_RW3, 0, Some("hint_flexible"));
    assert_eq!((r, flexible), (OK, base + PAGE * 2), "a flexible hint inside the reservation is used");
    t.write(flexible, 22);
    let q = t.query(direct, 0);
    assert!(q.direct && q.offset == phys);
    assert!(t.query(flexible, 0).flexible);
    let q = t.query(base + PAGE * 3, 0);
    assert!(!q.committed && q.prot == 0);
    // A hint over live mappings is only a hint: the mapping goes elsewhere.
    let (r, overlapping) = t.direct(base + PAGE, PAGE * 2, PROT_RW3, 0, phys + PAGE, 0, "hint_overlap");
    assert_eq!(r, OK);
    assert_ne!(overlapping, base + PAGE);
    assert_eq!((t.read(direct), t.read(flexible)), (21, 22));
    assert_eq!(t.munmap(overlapping, PAGE * 2), OK);
    assert_eq!(t.munmap(base, PAGE * 4), OK);
    assert_eq!(t.release(phys, PAGE * 3), OK);
}

#[test]
fn reserve_virtual_range_uses_a_free_hint() {
    let t = T::new();
    let (r, probe) = t.reserve(0, PAGE * 8, 0, 0);
    assert_eq!(r, OK);
    assert_eq!(t.munmap(probe, PAGE * 8), OK);
    let hint = probe + PAGE * 2;
    assert_eq!(t.reserve(hint, PAGE * 2, 0, 0), (OK, hint));
    let (r, above) = t.reserve(hint, PAGE * 2, 0, 0);
    assert_eq!(r, OK);
    assert!(above > hint, "an occupied hint searches upward");
    assert_eq!(t.munmap(above, PAGE * 2), OK);
    assert_eq!(t.munmap(hint, PAGE * 2), OK);
}

#[test]
fn release_direct_memory_rejects_malformed_ranges() {
    let t = T::new();
    let size = direct_size();
    for (start, len) in [
        (u64::MAX, PAGE),
        (0, 1),
        (0, PAGE - 1),
        (1, PAGE),
        (0, u64::MAX),
        (0, size + PAGE),
        (size - PAGE, PAGE * 2),
        (size, PAGE),
    ] {
        assert_eq!(t.release(start, len), EINVAL, "release({start:#x}, {len:#x})");
        assert_eq!(t.checked_release(start, len), EINVAL, "checked release({start:#x}, {len:#x})");
    }
}

#[test]
fn releasing_direct_memory_unmaps_every_view() {
    let t = T::new();
    let phys = t.alloc_ok(PAGE * 2, PAGE);
    let (r, mapped) = t.direct(0, PAGE * 2, PROT_RW3, 0, phys, 0, "released_view");
    assert_eq!(r, OK);
    assert_eq!(t.munmap(mapped + PAGE, PAGE), OK);
    let (r, alias) = t.direct(0, PAGE, PROT_RW3, 0, phys, 0, "released_alias");
    assert_eq!(r, OK);
    assert_eq!(t.release(phys, PAGE), OK);
    for view in [mapped, alias] {
        t.expect_unmapped(view);
        let (r, b) = t.query_raw(view, 1);
        if r == OK {
            assert_ne!(Q::from(&b).start, view);
        }
    }
    assert_eq!(t.release(phys + PAGE, PAGE), OK);
}

#[test]
fn virtual_query_reports_current_extents_of_split_flexible_ranges() {
    let t = T::new();
    let f = t.flex_ok(PAGE * 3, PROT_RW3, "vq_split");
    t.write(f, 17);
    t.write(f + PAGE * 2, 29);
    assert_eq!(t.call(cb_mprotect, &[f + PAGE, PAGE, PROT_CPU_READ as u64]).0, OK);
    for i in 0..3 {
        let q = t.query(f + PAGE * i, 0);
        assert_eq!((q.start, q.end), (f + PAGE * i, f + PAGE * (i + 1)));
        assert_eq!(q.prot, if i == 1 { PROT_CPU_READ } else { PROT_RW3 });
        assert!(q.committed && q.flexible && !q.direct);
    }
    assert_eq!(t.munmap(f + PAGE, PAGE), OK);
    for i in [0, 2] {
        let q = t.query(f + PAGE * i, 0);
        assert_eq!((q.start, q.end), (f + PAGE * i, f + PAGE * (i + 1)));
    }
    let next = t.query(f + PAGE, 1);
    assert_eq!((next.start, next.end, next.prot), (f + PAGE * 2, f + PAGE * 3, PROT_RW3));
    assert_eq!((t.read(f), t.read(f + PAGE * 2)), (17, 29));
    assert_eq!(t.munmap(f, PAGE), OK);
    assert_eq!(t.munmap(f + PAGE * 2, PAGE), OK);
}

#[test]
fn virtual_query_reports_non_readable_and_partly_unmapped_mmap_ranges() {
    let t = T::new();
    let m = anon(PAGE);
    assert_eq!(t.call(cb_posix_mprotect, &[m, PAGE, 0]).0, 0);
    let q = t.query(m, 0);
    assert_eq!((q.start, q.end, q.prot), (m, m + PAGE, 0));
    assert_eq!(t.call(cb_posix_munmap, &[m, PAGE]).0, 0);

    let m = anon(PAGE * 3);
    let q = t.query(m, 0);
    assert_eq!((q.start, q.end), (m, m + PAGE * 3));
    assert_eq!(t.call(cb_posix_munmap, &[m + PAGE, PAGE]).0, 0);
    let q = t.query(m, 0);
    assert_eq!((q.start, q.end), (m, m + PAGE));
    t.expect_unmapped(m + PAGE);
    let q = t.query(m + PAGE, 1);
    assert_eq!((q.start, q.end), (m + PAGE * 2, m + PAGE * 3));
    let q = t.query(m + PAGE * 2, 0);
    assert_eq!((q.start, q.end), (m + PAGE * 2, m + PAGE * 3));
    assert_eq!(t.call(cb_posix_munmap, &[m, PAGE]).0, 0);
    assert_eq!(t.call(cb_posix_munmap, &[m + PAGE * 2, PAGE]).0, 0);
}

#[test]
fn mmap_flags_alignment_and_kernel_variant() {
    let t = T::new();
    const SUPER: u64 = 2 * 1024 * 1024;
    assert_eq!(posix_mmap(0, 0, PROT_RW3, MAP_PRIVATE_ANON, -1, 0), Err(ERRNO_EINVAL));
    assert_eq!(posix_mmap(0, PAGE, 8, MAP_PRIVATE_ANON, -1, 0), Err(ERRNO_EINVAL), "only CPU protections");
    assert_eq!(posix_mmap(0, PAGE, PROT_RW3, MAP_PRIVATE_ANON, 3, 0), Err(ERRNO_EINVAL));
    assert_eq!(posix_mmap(0, PAGE, PROT_RW3, MAP_PRIVATE_ANON, -1, PAGE), Err(ERRNO_EINVAL));
    assert_eq!(posix_mmap(0, PAGE, PROT_RW3, 0x2, 3, 0), Err(ERRNO_ENOTSUP), "file mappings");
    assert_eq!(posix_mmap(0, PAGE, PROT_RW3, 0x200_1002, -1, 0), Err(ERRNO_EINVAL), "MAP_ALIGNED below a page");
    let a = posix_mmap(0, SUPER + PAGE, PROT_RW3, 0x100_1002, -1, 0).expect("MAP_ALIGNED_SUPER");
    assert_eq!(a % SUPER, 0);
    t.write(a, 1);
    t.write(a + SUPER + PAGE - 8, 2);
    let q = t.query(a, 0);
    assert_eq!((q.start, q.end, q.prot, q.committed), (a, a + SUPER + PAGE, PROT_RW3, true));
    assert_eq!(t.call(cb_posix_munmap, &[a, SUPER + PAGE]).0, 0);
    // POSIX mmap reports failure as -1.
    assert_eq!(t.call(cb_mmap, &[0, 0, PROT_RW3 as u64, MAP_PRIVATE_ANON, u64::MAX, 0]).0, -1);

    // sceKernelMmap returns the address through its seventh argument.
    let res = t.base + 0x60;
    let rsp = t.base + 0x400;
    let kernel_mmap = |len: u64| {
        let mut ctx = t.ctx();
        ctx.write_u64(res, 0);
        ctx.write_u64(rsp + 8, res);
        for (r, v) in [CpuRegister::Rdi, CpuRegister::Rsi, CpuRegister::Rdx, CpuRegister::Rcx, CpuRegister::R8, CpuRegister::R9]
            .iter()
            .zip([0, len, PROT_RW3 as u64, MAP_PRIVATE_ANON, u64::MAX, 0])
        {
            ctx.set(*r, v);
        }
        ctx.set(CpuRegister::Rsp, rsp);
        (cb_kernel_mmap(&mut ctx), ctx.read_u64(res).unwrap())
    };
    let (r, k) = kernel_mmap(PAGE);
    assert_eq!(r, OK);
    assert!(k != 0 && t.query(k, 0).committed);
    assert_eq!(t.munmap(k, PAGE), OK);
    assert_eq!(kernel_mmap(0), (EINVAL, 0));
}

#[test]
fn madvise_and_posix_unmap_protect_contracts() {
    let t = T::new();
    let m = anon(PAGE * 3);
    t.write(m, 42);
    t.write(m + PAGE * 2, 73);
    for advice in 0..=9 {
        assert_eq!(posix_madvise(m, PAGE * 3, advice), Ok(0), "advice {advice}");
    }
    assert_eq!((t.read(m), t.read(m + PAGE * 2)), (42, 73), "advice never changes contents");
    assert_eq!(posix_madvise(m + 1, 0, 4), Ok(0));
    assert_eq!(posix_madvise(m, PAGE, 11), Err(ERRNO_EINVAL));
    assert_eq!(posix_madvise(m, PAGE, -1), Err(ERRNO_EINVAL));
    assert_eq!(posix_madvise(m, 0, 11), Err(ERRNO_EINVAL));
    assert_eq!(posix_madvise(m, u64::MAX, 0), Err(ERRNO_EINVAL));
    assert_eq!(posix_madvise(0x8000_0000_0000, 1, 0), Err(ERRNO_EINVAL));
    assert_eq!(posix_madvise(0x7fff_ffff_c000, 0x4000, 4), Ok(0));
    assert_eq!(t.call(cb_madvise, &[m, PAGE, 11]).0, -1);
    assert_eq!(t.call(cb_posix_munmap, &[m, 0]).0, -1, "zero length");
    assert_eq!(t.read(m), 42);
    assert_eq!(t.call(cb_posix_mprotect, &[m, 0, 0]).0, 0, "zero length is a no-op");
    assert_eq!(t.call(cb_posix_mprotect, &[0, PAGE, 1]).0, -1);
    assert_eq!(t.call(cb_posix_munmap, &[m, PAGE * 3]).0, 0);
    assert_eq!(t.call(cb_posix_munmap, &[m, PAGE]).0, 0, "nothing mapped is not an error");
    assert_eq!(posix_madvise(m, PAGE, 4), Ok(0), "advice does not need a mapping");
    assert_eq!(t.call(cb_posix_mprotect, &[m, PAGE, 1]).0, -1);
}

/// The kernel's `munmap` rounds the start down and the end up to page
/// boundaries instead of rejecting an unaligned address; only a zero
/// length is EINVAL (13.60 syscall 73).
#[test]
fn posix_munmap_rounds_the_range_like_the_kernel() {
    let t = T::new();
    let m = anon(PAGE * 2);
    t.write(m, 42);
    t.write(m + PAGE, 73);
    assert_eq!(t.call(cb_posix_munmap, &[m + 1, PAGE]).0, 0, "the start rounds down, the end up");
    t.expect_unmapped(m);
    t.expect_unmapped(m + PAGE);
    assert_eq!(t.call(cb_posix_munmap, &[m, 0]).0, -1, "a zero length stays EINVAL");
    assert_eq!(t.call(cb_posix_munmap, &[u64::MAX, PAGE]).0, -1, "range overflow");
}

/// sceKernelMunmap is the munmap system call with an SCE error code: the
/// length rounds up to whole pages (a sub-page tail used to stay behind as
/// a range whose host pages were gone), the start rounds down.
#[test]
fn sce_munmap_rounds_the_range_like_the_kernel() {
    let t = T::new();
    let before = t.available_flexible();
    let f = t.flex_ok(PAGE * 3, PROT_RW3, "sce_munmap_unaligned");
    t.write(f + PAGE * 2, 7);
    assert_eq!(t.call(cb_munmap, &[f, PAGE + 0xa0]).0, OK, "the end rounds up");
    t.expect_unmapped(f);
    t.expect_unmapped(f + PAGE);
    t.expect_unmapped(f + PAGE + 0xa0);
    let q = t.query(f + PAGE * 2, 0);
    assert_eq!((q.start, q.end), (f + PAGE * 2, f + PAGE * 3), "the rest stays one whole page");
    assert_eq!(t.read(f + PAGE * 2), 7);
    assert_eq!(t.available_flexible(), before - PAGE, "two pages returned to the budget");
    assert_eq!(t.call(cb_munmap, &[f + PAGE * 2 + 8, 8]).0, OK, "the start rounds down");
    t.expect_unmapped(f + PAGE * 2);
    assert_eq!(t.available_flexible(), before);
    assert_eq!(t.call(cb_munmap, &[f, 0]).0, EINVAL, "a zero length is EINVAL");
    assert_eq!(t.call(cb_munmap, &[u64::MAX, PAGE]).0, EINVAL, "range overflow");
}

#[test]
fn msync_contracts() {
    let t = T::new();
    let m = anon(PAGE * 3);
    t.write(m, 1);
    for (addr, len, flags) in [(m, PAGE * 3, 0), (m, PAGE * 3, 1), (m + 100, 10, 0), (m + PAGE - 1, 2, 0), (m + PAGE, 0, 0), (m, PAGE, 4)] {
        assert_eq!(posix_msync(addr, len, flags), Ok(0), "msync({addr:#x}, {len:#x}, {flags})");
    }
    // The console refuses MS_INVALIDATE with EBUSY before any other check
    // (measured on hardware: the combined flag fails the same way).
    assert_eq!(posix_msync(m, PAGE, 2), Err(ERRNO_EBUSY), "MS_INVALIDATE");
    assert_eq!(posix_msync(m, PAGE, 3), Err(ERRNO_EBUSY), "MS_ASYNC | MS_INVALIDATE");
    let top = u64::MAX & !(PAGE - 1);
    assert_eq!(posix_msync(top, PAGE * 2, 0), Err(ERRNO_EINVAL));
    assert_eq!(t.call(cb_posix_munmap, &[m, PAGE * 3]).0, 0);
    for (addr, len, flags) in [(m, PAGE, 0), (m + PAGE, PAGE * 3, 1), (m + PAGE, 0, 0)] {
        assert_eq!(posix_msync(addr, len, flags), Err(ERRNO_ENOMEM), "msync({addr:#x}, {len:#x}, {flags})");
    }
    // MS_INVALIDATE fails with EBUSY even unmapped (checked first).
    assert_eq!(posix_msync(m, PAGE, 2), Err(ERRNO_EBUSY));
    assert_eq!(t.call(cb_msync, &[m, PAGE, 0]).0, -1);
    assert_eq!(t.call(cb_kernel_msync, &[m, PAGE, 0]).0, ENOMEM);
    let (r, reserved) = t.reserve(0, PAGE * 2, 0, 0);
    assert_eq!(r, OK);
    assert_eq!(posix_msync(reserved, PAGE * 2, 0), Ok(0), "reservations are mapped");
    assert_eq!(posix_msync(reserved, 0, 1), Ok(0));
    assert_eq!(t.munmap(reserved, PAGE * 2), OK);
}

#[test]
fn mlock_and_munlock_follow_the_guest_mapping_state() {
    let t = T::new();
    let len = 0x10000;
    let f = t.flex_ok(len, PROT_RW3, "mlock_flexible");
    assert_eq!(kernel_mlock(f, 0), OK);
    assert_eq!(kernel_mlock(f + 1, len - PAGE), OK);
    assert_eq!(kernel_mlock(f, len), OK);
    assert_eq!(kernel_mlock(f, len), OK, "locking twice");
    assert_eq!(kernel_munlock(f, 0), OK);
    assert_eq!(kernel_munlock(f + 1, len - PAGE), OK);
    assert_eq!(kernel_munlock(f, len), OK);
    let top = u64::MAX - PAGE + 1;
    assert_eq!(kernel_mlock(top, PAGE * 2), EINVAL);
    assert_eq!(kernel_munlock(top, PAGE * 2), EINVAL);
    assert_eq!(t.call(cb_posix_mlock, &[top, PAGE * 2]).0, -1);
    let (r, reserved) = t.reserve(0, PAGE, 0, 0);
    assert_eq!(r, OK);
    assert_eq!(kernel_mlock(reserved, PAGE), ENOMEM, "a reservation holds no memory to lock");
    assert_eq!(kernel_munlock(reserved, PAGE), OK, "but it is mapped");
    assert_eq!(t.call(cb_posix_munlock, &[reserved, PAGE]).0, 0);
    assert_eq!(t.munmap(reserved, PAGE), OK);
    assert_eq!(t.munmap(f, len), OK);
    assert_eq!(kernel_mlock(f, PAGE), ENOMEM);
    assert_eq!(kernel_munlock(f, PAGE), ENOMEM);
    assert_eq!(t.call(cb_kernel_munlock, &[f, PAGE]).0, ENOMEM);
    // Memory outside the map that the guest range registry tracks (loader
    // images) counts as committed.
    assert_eq!(kernel_mlock(t.base + 1, 0x100), OK);
    assert_eq!(kernel_munlock(t.base + 1, 0x100), OK);
}

/// sceKernelVirtualQuery writes the SysV layout of the console's
/// SceKernelVirtualQueryInfo: the flag bits in byte 32, the name from byte
/// 33, the GPU mask id at 65 and nothing past the 72 bytes.
#[test]
fn virtual_query_wire_layout() {
    let t = T::new();
    let name = "query wire layout";
    let m = t.flex_ok(PAGE, 3, name);
    let info = t.base + 0x200;
    let ctx = t.ctx();
    ctx.write_bytes(info - 16, &[0xa5; 16]);
    ctx.write_bytes(info + 72, &[0x5a; 16]);
    assert_eq!(t.call(cb_virtual_query, &[m, 0, info, 72]).0, OK);
    let b = ctx.read_bytes(info, 72).unwrap();
    assert_eq!(u64::from_le_bytes(b[0..8].try_into().unwrap()), m);
    assert_eq!(u64::from_le_bytes(b[8..16].try_into().unwrap()), m + PAGE);
    assert_eq!(i32::from_le_bytes(b[24..28].try_into().unwrap()), 3);
    assert_eq!(b[32], 0x11, "flexible and committed");
    assert_eq!(&b[33..33 + name.len()], name.as_bytes());
    assert_eq!(b[33 + name.len()], 0);
    assert!(b[65..72].iter().all(|&x| x == 0));
    assert_eq!(ctx.read_bytes(info - 16, 16).unwrap(), vec![0xa5; 16]);
    assert_eq!(ctx.read_bytes(info + 72, 16).unwrap(), vec![0x5a; 16]);
    assert_eq!(t.munmap(m, PAGE), OK);
}

/// The queries report the read bit of every write bit (PS5 FW 13.60):
/// sceKernelVirtualQuery and sceKernelQueryMemoryProtection agree.
#[test]
fn reported_protection_adds_read_for_write() {
    let t = T::new();
    let base = t.flex_ok(PAGE, 0x2, "reported_prot");
    let s = t.base + 0x80;
    let check = |want: i32| {
        assert_eq!(t.query(base, 0).prot, want);
        let (r, ctx) = t.call(cb_query_memory_protection, &[base, 0, 0, s]);
        assert_eq!(r, OK);
        assert_eq!(ctx.read_u32(s), Some(want as u32));
    };
    check(0x3);
    for (req, want) in [
        (0x1u64, 0x1),
        (0x2, 0x3),
        (0x10, 0x10),
        (0x12, 0x13),
        (0x20, 0x20),
        (0x22, 0x23),
        (0x40, 0x40),
        (0x80, 0xc0),
        (0x82, 0xc3),
        (0x100, 0x100),
        (0x200, 0x300),
        (0x202, 0x303),
        (0x220, 0x320),
        (0x2a2, 0x3e3),
        (0xf2, 0xf3),
        (0x3f2, 0x3f3),
        (0xffff_fff2, 0x3f3),
    ] {
        assert_eq!(t.call(cb_mprotect, &[base, PAGE, req]).0, OK, "{req:#x}");
        check(want);
    }
    assert_eq!(t.munmap(base, PAGE), OK);
}

/// sceKernelMprotect: a zero length is 0, bits outside 0x3F7 are dropped,
/// holes and reservations inside the range are skipped while the committed
/// parts are protected, and an unmapped start is EINVAL.
#[test]
fn mprotect_skips_holes_and_reservations() {
    let t = T::new();
    let (r, base) = t.reserve(0, PAGE * 4, 0, 0);
    assert_eq!(r, OK);
    assert_eq!(t.flex(base, PAGE, PROT_CPU_RW, MAP_FIXED, Some("mprotect_a")).0, OK);
    assert_eq!(t.flex(base + PAGE * 2, PAGE, PROT_CPU_RW, MAP_FIXED, Some("mprotect_b")).0, OK);
    assert_eq!(t.call(cb_mprotect, &[base, 0, 0]).0, OK, "zero length");
    assert_eq!(t.call(cb_mprotect, &[base + 1, 0, 0]).0, OK, "zero length at an unaligned address");
    assert_eq!(t.query(base, 0).prot, reported_protection(PROT_CPU_RW));
    // 0xfffffff2: CPU write plus undefined bits.
    assert_eq!(t.call(cb_mprotect, &[base, PAGE * 4, 0xffff_fff2]).0, OK);
    assert_eq!(t.query(base, 0).prot, 0x3F3);
    assert_eq!(t.query(base + PAGE * 2, 0).prot, 0x3F3);
    let q = t.query(base + PAGE, 0);
    assert!(!q.committed && q.prot == 0, "a reservation keeps its protection");
    assert_eq!(t.call(cb_mprotect, &[base, PAGE * 3, PROT_CPU_RD as u64]).0, OK);
    assert_eq!(t.query(base, 0).prot, PROT_CPU_RD);
    assert_eq!(t.read(base), 0, "readable after the change");
    assert_eq!(t.query(base + PAGE * 2, 0).prot, PROT_CPU_RD);
    // A hole inside the range.
    assert_eq!(t.munmap(base + PAGE, PAGE), OK);
    assert_eq!(t.call(cb_mprotect, &[base, PAGE * 3, PROT_CPU_RW as u64]).0, OK);
    assert_eq!(t.query(base + PAGE * 2, 0).prot, reported_protection(PROT_CPU_RW));
    // An unmapped start.
    assert_eq!(t.call(cb_mprotect, &[base + PAGE, PAGE * 2, PROT_CPU_RW as u64]).0, EINVAL);
    assert_eq!(t.munmap(base, PAGE * 4), OK);
    assert_eq!(t.call(cb_mprotect, &[base, PAGE, PROT_CPU_RW as u64]).0, EINVAL);
}

/// sceKernelMtypeprotect gives the direct mappings of the range the memory
/// type; a zero length is 0.
#[test]
fn mtypeprotect_changes_the_memory_type() {
    let t = T::new();
    let phys = t.alloc_ok(PAGE * 2, PAGE);
    let (r, m) = t.direct(0, PAGE * 2, PROT_CPU_RW, 0, phys, PAGE, "mtype");
    assert_eq!(r, OK);
    assert_eq!(t.query(m, 0).mtype, MTYPE_C);
    assert_eq!(t.call(cb_mtypeprotect, &[m, 0, 3, PROT_CPU_RD as u64]).0, OK);
    assert_eq!(t.query(m, 0).prot, reported_protection(PROT_CPU_RW));
    assert_eq!(t.call(cb_mtypeprotect, &[m + PAGE, PAGE, 3, PROT_CPU_RD as u64]).0, OK);
    let q = t.query(m + PAGE, 0);
    assert_eq!((q.mtype, q.prot, q.start), (3, PROT_CPU_RD, m + PAGE));
    assert_eq!(t.query(m, 0).mtype, MTYPE_C);
    assert_eq!(t.munmap(m, PAGE * 2), OK);
    assert_eq!(t.release(phys, PAGE * 2), OK);
}

/// munmap over a range with holes unmaps what is mapped and skips the
/// holes; over nothing mapped it is 0; a zero length is EINVAL.
#[test]
fn munmap_over_unmapped_pages() {
    let t = T::new();
    let m = anon(PAGE * 3);
    t.write(m, 1);
    t.write(m + PAGE * 2, 2);
    assert_eq!(t.call(cb_posix_munmap, &[m + PAGE, PAGE]).0, 0);
    assert_eq!(t.call(cb_posix_munmap, &[m, PAGE * 3]).0, 0);
    t.expect_unmapped(m);
    t.expect_unmapped(m + PAGE * 2);
    assert_eq!(t.call(cb_posix_munmap, &[m, PAGE * 3]).0, 0);
    assert_eq!(t.call(cb_munmap, &[m, PAGE * 3]).0, OK);
    assert_eq!(t.call(cb_munmap, &[m, 0]).0, EINVAL);
}

/// mincore follows FreeBSD 11's kern_mincore: the address rounds down to its
/// page, committed pages report 1 and reserved ones 0, an unmapped page or
/// an overflowing range is ENOMEM, then a null vector EFAULT; a zero length
/// only checks the start page.
#[test]
fn mincore_reports_committed_pages() {
    let t = T::new();
    let (r, base) = t.reserve(0, PAGE * 3, 0, 0);
    assert_eq!(r, OK);
    assert_eq!(t.flex(base + PAGE, PAGE, PROT_CPU_RW, MAP_FIXED, Some("mincore")).0, OK);
    let vec = t.base + 0x300;
    let ctx = t.ctx();
    ctx.write_bytes(vec, &[0xee; 4]);
    assert_eq!(posix_mincore(&ctx, base + 5, PAGE * 3 - 5, vec), Ok(0));
    assert_eq!(ctx.read_bytes(vec, 4).unwrap(), vec![0, 1, 0, 0xee]);
    assert_eq!(posix_mincore(&ctx, base, PAGE, 0), Err(14), "null vector");
    assert_eq!(posix_mincore(&ctx, base, 0, 0), Ok(0), "zero length checks the page only");
    assert_eq!(posix_mincore(&ctx, u64::MAX - 10, 100, vec), Err(ERRNO_ENOMEM));
    assert_eq!(t.munmap(base, PAGE * 3), OK);
    assert_eq!(posix_mincore(&ctx, base, PAGE, vec), Err(ERRNO_ENOMEM), "unmapped");
    assert_eq!(posix_mincore(&ctx, base, 0, 0), Err(ERRNO_ENOMEM));
    // MADV_PROTECT ignores the range.
    assert_eq!(posix_madvise(u64::MAX, u64::MAX, 10), Ok(0));
}

/// AMM command execution against the memory manager: MultiMap aliases the
/// source pages, Remap moves them (the source is left unmapped), the
/// protect commands merge under their mask, MapAsPrt drops the backing and
/// AllocatePaForPrt backs a PRT range from the automatic pool.
#[test]
fn amm_remap_multimap_protect_and_prt() {
    use crate::ampr::amm;
    use crate::ampr::command_buffer::{AmmKind, AmmMap};
    let t = T::new();
    let base = EXTENDED_MEMORY_BASE + 0x3000_0000;
    let cmd = |kind, va, source, size, prot, mask, mtype| AmmMap { kind, va, source, size, prot, mask, mtype, ..Default::default() };
    let phys = t.alloc_ok(PAGE * 2, PAGE);
    let direct = AmmMap { kind: AmmKind::MapDirect, va: base, dmem_offset: phys, size: PAGE * 2, prot: PROT_CPU_RW, ..Default::default() };
    assert_eq!(amm::execute(&direct), OK);
    t.write(base + PAGE, 0x414d_4d5f_5041_4745);

    let alias = base + 0x10_0000;
    assert_eq!(amm::execute(&cmd(AmmKind::MultiMap, alias, base, PAGE * 2, PROT_CPU_RW, 0, 0)), OK);
    assert_eq!(t.read(alias + PAGE), 0x414d_4d5f_5041_4745, "the alias shares the pages");
    assert_eq!(t.query(alias, 0).offset, phys);
    assert_eq!(amm::execute(&cmd(AmmKind::MultiMap, base + PAGE, base, PAGE * 2, PROT_CPU_RW, 0, 0)), EINVAL, "onto its own source");
    assert_eq!(amm::execute(&cmd(AmmKind::Unmap, alias, 0, PAGE * 2, 0, 0, 0)), OK);

    let moved = base + 0x20_0000;
    assert_eq!(amm::execute(&cmd(AmmKind::Remap, moved, base, PAGE * 2, PROT_CPU_RD, 0, 0)), OK);
    t.expect_unmapped(base);
    assert_eq!(t.read(moved + PAGE), 0x414d_4d5f_5041_4745);
    let q = t.query(moved, 0);
    assert_eq!((q.offset, q.prot, q.mtype), (phys, PROT_CPU_RD, MTYPE_C));
    assert_eq!(amm::execute(&cmd(AmmKind::Remap, alias, base, PAGE * 2, PROT_CPU_RW, 0, 0)), EINVAL, "unmapped source");

    assert_eq!(amm::execute(&cmd(AmmKind::ModifyProtect, moved, 0, PAGE * 2, PROT_CPU_RW, PROT_CPU_RW, 0)), OK);
    assert_eq!(t.query(moved, 0).prot, reported_protection(PROT_CPU_RW) | PROT_CPU_RD);
    assert_eq!(amm::execute(&cmd(AmmKind::ModifyMtypeProtect, moved, 0, PAGE, 0, PROT_CPU_RW, 3)), OK);
    let q = t.query(moved, 0);
    assert_eq!((q.prot, q.mtype, q.end), (PROT_CPU_RD, 3, moved + PAGE), "only the masked bit and the first page change");
    assert_eq!(t.query(moved + PAGE, 0).prot, reported_protection(PROT_CPU_RW) | PROT_CPU_RD);
    assert_eq!(amm::execute(&cmd(AmmKind::ModifyProtect, base, 0, PAGE, PROT_CPU_RW, PROT_CPU_RW, 0)), EINVAL, "not mapped");

    // PRT: backing from the automatic pool.
    let prt = base + 0x30_0000;
    let pool = t.alloc_ok(PAGE * 2, PAGE);
    amm::give_automatic(pool, PAGE * 2);
    let allocate = |prot| cmd(AmmKind::AllocatePaForPrt, prt, 0, PAGE * 2, prot, 1019, MTYPE_C);
    assert_eq!(amm::execute(&allocate(PROT_CPU_RW)), EINVAL, "not a PRT range");
    assert_eq!(amm::execute(&cmd(AmmKind::MapAsPrt, prt, 0, PAGE * 2, 0, 0, 0)), OK);
    t.expect_unmapped(prt);
    assert_eq!(amm::execute(&allocate(PROT_CPU_RW)), OK);
    t.write(prt + PAGE, 7);
    assert_eq!(t.read(prt + PAGE), 7);
    assert!((pool..pool + PAGE * 2).contains(&t.query(prt, 0).offset));
    assert_eq!(amm::execute(&allocate(PROT_CPU_RD)), OK, "backed pages take the new protection");
    assert_eq!(t.query(prt, 0).prot, PROT_CPU_RD);
    assert_eq!(amm::execute(&cmd(AmmKind::Unmap, prt, 0, PAGE * 2, 0, 0, 0)), OK);
    assert_eq!(amm::execute(&allocate(PROT_CPU_RW)), EINVAL, "unmap ends the PRT range");

    assert_eq!(amm::execute(&cmd(AmmKind::Unmap, moved, 0, PAGE * 2, 0, 0, 0)), OK);
    assert_eq!(t.release(phys, PAGE * 2), OK);
    amm::withdraw_automatic(pool, PAGE * 2);
    assert_eq!(t.release(pool, PAGE * 2), OK);
}
