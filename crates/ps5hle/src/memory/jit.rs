// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! JIT shared memory: sceKernelJitCreateSharedMemory makes a shared memory
//! object behind a descriptor, sceKernelJitCreateAliasOfSharedMemory a
//! second descriptor with its own maximum protection, and
//! sceKernelJitMapSharedMemory maps it (one writable view, one executable
//! view of the same pages). Closing a descriptor forgets it; mappings stay.

#[cfg(windows)]
#[allow(unused_imports)]
use crate::host::c as libc;
use super::*;
use std::collections::HashMap;
use std::sync::Arc;

const CPU_READ: i32 = 0x1;
const CPU_WRITE: i32 = 0x2;
const CPU_EXEC: i32 = 0x4;
const CPU_PROT: i32 = CPU_READ | CPU_WRITE | CPU_EXEC;
const EBADF: HleResult = kerr(9);

/// Writing or executing implies reading.
fn granted(prot: i32) -> i32 {
    if prot & (CPU_WRITE | CPU_EXEC) != 0 { prot | CPU_READ } else { prot }
}

struct JitObject {
    memfd: std::fs::File,
    len: u64,
}

static JIT: Mutex<Option<HashMap<i32, (Arc<JitObject>, i32)>>> = Mutex::new(None);

fn with_jit<R>(f: impl FnOnce(&mut HashMap<i32, (Arc<JitObject>, i32)>) -> R) -> R {
    let mut g = JIT.lock().unwrap_or_else(|e| e.into_inner());
    f(g.get_or_insert_with(HashMap::new))
}

/// close() hook: true when `fd` was a JIT shared memory descriptor.
pub(crate) fn close(fd: i32) -> bool {
    let closed = with_jit(|m| m.remove(&fd).is_some());
    if closed {
        crate::filesystem::release_fd(fd);
    }
    closed
}

/// True when `fd` is a JIT shared memory descriptor.
pub(crate) fn is_jit(fd: i32) -> bool {
    with_jit(|m| m.contains_key(&fd))
}

/// sceKernelJitCreateSharedMemory(name, len, maxProt, int* fd)
fn cb_create(ctx: &mut CpuContext) -> HleResult {
    let (len, max_prot, out) = (ctx.rsi(), ctx.rdx() as i32, ctx.rcx());
    if out == 0 || len == 0 || len % PAGE_SIZE != 0 || max_prot & !CPU_PROT != 0 {
        return EINVAL;
    }
    let Ok(memfd) = crate::host::fs::memfd("rps5e jit shared memory") else { return ENOMEM };
    if len > i64::MAX as u64 || memfd.set_len(len).is_err() {
        return ENOMEM;
    }
    let fd = crate::filesystem::reserve_fd();
    with_jit(|m| m.insert(fd, (Arc::new(JitObject { memfd, len }), max_prot)));
    if !ctx.write_u32(out, fd as u32) {
        close(fd);
        return EFAULT;
    }
    OK
}

/// sceKernelJitCreateAliasOfSharedMemory(fd, maxProt, int* alias): no
/// more access than the source grants (EACCES).
fn cb_create_alias(ctx: &mut CpuContext) -> HleResult {
    let (fd, max_prot, out) = (ctx.rdi() as i32, ctx.rsi() as i32, ctx.rdx());
    if out == 0 || max_prot & !CPU_PROT != 0 {
        return EINVAL;
    }
    let Some((object, source_prot)) = with_jit(|m| m.get(&fd).cloned()) else { return EBADF };
    if granted(max_prot) & !granted(source_prot) != 0 {
        return EACCES;
    }
    let alias = crate::filesystem::reserve_fd();
    with_jit(|m| m.insert(alias, (object, max_prot)));
    if !ctx.write_u32(out, alias as u32) {
        close(alias);
        return EFAULT;
    }
    OK
}

/// sceKernelJitMapSharedMemory(fd, prot, void** start): `*start` is a
/// placement hint; the mapping address is stored there.
fn cb_map(ctx: &mut CpuContext) -> HleResult {
    let (fd, prot, start) = (ctx.rdi() as i32, ctx.rsi() as i32, ctx.rdx());
    if start == 0 || prot & !CPU_PROT != 0 {
        return EINVAL;
    }
    let Some((object, max_prot)) = with_jit(|m| m.get(&fd).cloned()) else { return EBADF };
    if granted(prot) & !granted(max_prot) != 0 {
        return EACCES;
    }
    let Some(hint) = ctx.read_u64(start) else { return EFAULT };
    match map_shared_host_fd(hint, object.len, prot, crate::host::fs::as_raw_fd(&object.memfd), "jit") {
        Ok(addr) => {
            if ctx.write_u64(start, addr) { OK } else { EFAULT }
        }
        Err(e) => e,
    }
}

pub(super) const JIT_EXPORTS: &[(&str, &str, crate::registry::HleCallback)] = &[
    ("avvJ3J0H0EY", "sceKernelJitCreateSharedMemory", cb_create),
    ("MR221Mwo0Pc", "sceKernelJitCreateAliasOfSharedMemory", cb_create_alias),
    ("YKT49TOLQWs", "sceKernelJitMapSharedMemory", cb_map),
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CpuRegister;
    use ps5vm::{MemProtect, VirtualMemory};

    const BASE: u64 = 0x7E65_0000_0000;

    fn call(ctx: &mut CpuContext, cb: crate::registry::HleCallback, args: &[u64]) -> HleResult {
        let regs = [CpuRegister::Rdi, CpuRegister::Rsi, CpuRegister::Rdx, CpuRegister::Rcx];
        for (r, v) in regs.iter().zip(args.iter().chain(std::iter::repeat(&0))) {
            ctx.set(*r, *v);
        }
        cb(ctx)
    }

    /// A writable and an executable view of one object share their bytes;
    /// aliases and views cannot exceed the granted protection.
    #[test]
    fn shared_views_and_protection() {
        let _serial = super::super::tests::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let vm = VirtualMemory::new();
        vm.map(BASE, 0x1000, MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let (fd_out, alias_out, start) = (BASE, BASE + 8, BASE + 16);
        assert_eq!(call(&mut ctx, cb_create, &[0, 0x1000, 7, fd_out]), EINVAL, "not a 16 KiB multiple");
        assert_eq!(call(&mut ctx, cb_create, &[0, 0x4000, 8, fd_out]), EINVAL);
        assert_eq!(call(&mut ctx, cb_create, &[0, 0x4000, 3, 0]), EINVAL);
        assert_eq!(call(&mut ctx, cb_create, &[0, 0x4000, CPU_READ as u64 | CPU_WRITE as u64, fd_out]), OK);
        let fd = ctx.read_u32(fd_out).unwrap() as u64;
        assert!(is_jit(fd as i32));
        assert_eq!(call(&mut ctx, cb_create_alias, &[fd, 7, alias_out]), EACCES);
        assert_eq!(call(&mut ctx, cb_create_alias, &[fd, 1, alias_out]), OK);
        let alias = ctx.read_u32(alias_out).unwrap() as u64;
        assert_eq!(call(&mut ctx, cb_create_alias, &[0x0FFF_FFF0, 1, alias_out]), EBADF);
        ctx.write_u64(start, 0);
        assert_eq!(call(&mut ctx, cb_map, &[alias, 3, start]), EACCES);
        assert_eq!(call(&mut ctx, cb_map, &[fd, 3, start]), OK);
        let rw = ctx.read_u64(start).unwrap();
        ctx.write_u64(start, 0);
        assert_eq!(call(&mut ctx, cb_map, &[alias, 1, start]), OK);
        let ro = ctx.read_u64(start).unwrap();
        assert!(rw != 0 && ro != 0 && rw != ro);
        // SAFETY: both views were just mapped for this test.
        unsafe {
            *(rw as *mut u32) = 0xC0DE_F00D;
            assert_eq!(*(ro as *const u32), 0xC0DE_F00D);
        }
        assert!(close(alias as i32) && close(fd as i32));
        assert!(!close(fd as i32));
        assert_eq!(super::unmap(rw, 0x4000), OK);
        assert_eq!(super::unmap(ro, 0x4000), OK);
    }

    #[test]
    fn nids_match_names() {
        for &(nid, name, _) in JIT_EXPORTS {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }
}
