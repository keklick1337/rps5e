// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Thread-local storage (x86-64 variant II) and per-thread control blocks.
//!
//! Every image with a non-empty `PT_TLS` segment gets a module id (main
//! executable first). Images loaded before the first thread exists get a
//! static offset: their block lives at `tp - offset` inside every thread
//! block and is seeded when the thread block is created. Images loaded
//! later (`sceKernelLoadStartModule`) use dynamic TLS: a thread's block is
//! allocated on its first `__tls_get_addr` for that module, seeded with the
//! module's initialization image, and freed with the thread (or the module).
//!
//! The initialization image is a snapshot of the `.tdata` bytes taken after
//! the image was relocated (`set_init_image`).
//!
//! A thread block region spans `[tp - STATIC_RESERVE, tp + TCB_SIZE)`:
//!
//! ```text
//!   tp - offset_n .. tp     static TLS blocks (init image + zero fill)
//!   tp + 0x00               self pointer
//!   tp + 0x08               DTV pointer (generation, count, block per id)
//!   tp + 0x10               pthread handle
//!   tp + 0x28               stack protector canary
//!   tp + 0x60               self pointer (second copy used by libc)
//!   tp + DTV_OFFSET         DTV
//!   tp + HANDLE_OFFSET      pthread object (64-bit thread id at +0)
//! ```

#[cfg(windows)]
#[allow(unused_imports)]
use crate::host::c as libc;
use crate::context::CpuContext;
use crate::registry::{HleRegistry, HleResult};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// Space reserved below the thread pointer for static TLS blocks.
pub const STATIC_RESERVE: u64 = 0x2_0000;
/// Space above the thread pointer: TCB, DTV and the pthread object.
pub const TCB_SIZE: u64 = 0x1_0000;
const DTV_OFFSET: u64 = 0x1000;
/// DTV entries available below the pthread object.
const DTV_MAX_ID: u64 = (HANDLE_OFFSET - DTV_OFFSET) / 8 - 2;
pub const HANDLE_OFFSET: u64 = 0x8000;
pub const HANDLE_SIZE: u64 = 0x1000;
pub const STACK_CANARY: u64 = 0xC0DE_C0DE_CAFE_BA00;

/// Thread pointer of slot `i`; slots are 16 MiB apart.
const TP_TOP: u64 = 0x6FFE_0000_0000;
const TP_STEP: u64 = 0x100_0000;
const MAX_SLOTS: usize = 1024;

#[derive(Clone, Debug)]
struct TlsModule {
    id: u64,
    memsz: u64,
    align: u64,
    /// Initialization image (`.tdata`); the rest of the block is zero.
    init: Vec<u8>,
    /// Guest address of the image in the loaded module (snapshot source).
    image: u64,
    filesz: u64,
    /// Static offset below the thread pointer; None for dynamic TLS.
    offset: Option<u64>,
}

struct TlsState {
    modules: Vec<TlsModule>,
    next_id: u64,
    next_offset: u64,
    /// A thread block exists: modules registered from now on are dynamic.
    threads_started: bool,
    slots: [bool; MAX_SLOTS],
    /// Dynamic blocks: (thread pointer, module id) -> (address, size).
    dynamic: BTreeMap<(u64, u64), (u64, u64)>,
}

static STATE: Mutex<TlsState> = Mutex::new(TlsState {
    modules: Vec::new(),
    next_id: 1,
    next_offset: 0,
    threads_started: false,
    slots: [false; MAX_SLOTS],
    dynamic: BTreeMap::new(),
});

/// Bumped whenever a module or a thread's dynamic block goes away, so
/// per-thread lookup caches drop stale entries.
static GENERATION: AtomicU64 = AtomicU64::new(0);

thread_local! {
    /// (generation, thread pointer, [(module id, block address, size)]).
    static CACHE: RefCell<(u64, u64, Vec<(u64, u64, u64)>)> = const { RefCell::new((0, 0, Vec::new())) };
}

fn lock() -> std::sync::MutexGuard<'static, TlsState> {
    STATE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Variant II static offset for a module following `previous` bytes.
fn static_offset(previous: u64, size: u64, align: u64, bias: u64) -> u64 {
    let align = align.max(1);
    let r = previous + size + align - 1;
    r - ((r + bias) & (align - 1))
}

/// Register an image's `PT_TLS` (before relocating it); returns
/// `(module_id, static_offset)` — the offset is 0 for a dynamic module — or
/// None when the image has no TLS.
pub fn register_module(align: u64, memsz: u64, filesz: u64, image: u64, vaddr: u64) -> Option<(u64, u64)> {
    if memsz == 0 {
        return None;
    }
    let mut st = lock();
    let id = st.next_id;
    if id > DTV_MAX_ID {
        log::error!("[tls] too many TLS modules");
        return None;
    }
    st.next_id += 1;
    let offset = if st.threads_started {
        None
    } else {
        let o = static_offset(st.next_offset, memsz, align, vaddr);
        if o > STATIC_RESERVE {
            // The module's TLS is not registered: the caller fails the load.
            log::error!("[tls] static TLS of {o:#x} bytes exceeds the {STATIC_RESERVE:#x} reservation");
            st.next_id -= 1;
            return None;
        }
        st.next_offset = o;
        Some(o)
    };
    st.modules.push(TlsModule { id, memsz, align, init: Vec::new(), image, filesz: filesz.min(memsz), offset });
    log::info!("[tls] module id={id} memsz={memsz:#x} filesz={filesz:#x} offset={offset:?}");
    Some((id, offset.unwrap_or(0)))
}

/// Snapshot module `id`'s initialization image from guest memory (call once
/// the image is relocated).
pub fn set_init_image(vm: &ps5vm::VirtualMemory, id: u64) {
    let mut st = lock();
    let Some(m) = st.modules.iter_mut().find(|m| m.id == id) else { return };
    let mut buf = vec![0u8; m.filesz as usize];
    if m.filesz != 0 && vm.read(m.image, &mut buf).is_err() {
        log::warn!("[tls] module {id}: cannot read the initialization image at {:#x}", m.image);
        return;
    }
    m.init = buf;
}

/// Forget module `id` (unloaded) and free its dynamic blocks.
pub fn unregister_module(id: u64) {
    let blocks: Vec<(u64, u64)> = {
        let mut st = lock();
        st.modules.retain(|m| m.id != id);
        let keys: Vec<(u64, u64)> = st.dynamic.keys().filter(|k| k.1 == id).copied().collect();
        keys.iter().filter_map(|k| st.dynamic.remove(k)).collect()
    };
    GENERATION.fetch_add(1, Ordering::Release);
    for (addr, size) in blocks {
        unmap(addr, size);
    }
}

/// Whether module `id` uses dynamic TLS.
pub fn is_dynamic(id: u64) -> bool {
    lock().modules.iter().any(|m| m.id == id && m.offset.is_none())
}

fn map_rw(addr: u64, size: u64, fixed: bool) -> Option<u64> {
    let flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | if fixed { libc::MAP_FIXED_NOREPLACE } else { 0 };
    let p = unsafe { libc::mmap(addr as *mut libc::c_void, size as usize, libc::PROT_READ | libc::PROT_WRITE, flags, -1, 0) };
    if p == libc::MAP_FAILED || (fixed && p as u64 != addr) {
        return None;
    }
    ps5vm::ranges::track(p as u64, size, ps5vm::MemProtect::RW);
    Some(p as u64)
}

fn unmap(addr: u64, size: u64) {
    unsafe { libc::munmap(addr as *mut libc::c_void, size as usize) };
    ps5vm::ranges::untrack(addr, size);
}

/// Allocate and seed a thread block; returns the thread pointer. The pthread
/// object (handle) is `tp + HANDLE_OFFSET`.
pub fn allocate_thread_block(vm: &ps5vm::VirtualMemory, thread_id: u32) -> Option<u64> {
    // A slot whose block cannot be mapped (the range is taken, e.g. by the
    // helper pages at 0x6FFC_/0x6FFD_0000_0000) stays marked used and the
    // next free one is tried, so it is never handed out again.
    let (tp, modules) = loop {
        let (tp, statics) = {
            let mut st = lock();
            st.threads_started = true;
            let slot = st.slots.iter().position(|used| !used)?;
            st.slots[slot] = true;
            let statics: Vec<TlsModule> = st.modules.iter().filter(|m| m.offset.is_some()).cloned().collect();
            (TP_TOP - slot as u64 * TP_STEP, statics)
        };
        if map_rw(tp - STATIC_RESERVE, STATIC_RESERVE + TCB_SIZE, true).is_some() {
            break (tp, statics);
        }
        log::warn!("[tls] thread block at {tp:#x} cannot be mapped; its slot is retired");
    };
    let handle = tp + HANDLE_OFFSET;
    let dtv = tp + DTV_OFFSET;
    let w = |a: u64, v: u64| vm.write(a, &v.to_le_bytes()).is_ok();
    w(tp, tp);
    w(tp + 0x08, dtv);
    w(tp + 0x10, handle);
    w(tp + 0x28, STACK_CANARY);
    w(tp + 0x60, tp);
    w(dtv, 1);
    w(dtv + 8, modules.iter().map(|m| m.id).max().unwrap_or(0));
    for m in &modules {
        let block = tp - m.offset.unwrap_or(0);
        if !m.init.is_empty() {
            let _ = vm.write(block, &m.init);
        } else if m.filesz != 0 {
            // Seeded before the snapshot was taken (main thread at start-up).
            let mut buf = vec![0u8; m.filesz as usize];
            if vm.read(m.image, &mut buf).is_ok() {
                let _ = vm.write(block, &buf);
            }
        }
        w(dtv + 8 + m.id * 8, block);
    }
    // The pthread object starts with the 64-bit thread id.
    w(handle, thread_id as u64);
    Some(tp)
}

/// Release a thread block allocated by `allocate_thread_block`, with the
/// thread's dynamic TLS blocks.
pub fn free_thread_block(tp: u64) {
    let blocks: Vec<(u64, u64)> = {
        let mut st = lock();
        let keys: Vec<(u64, u64)> = st.dynamic.keys().filter(|k| k.0 == tp).copied().collect();
        let blocks = keys.iter().filter_map(|k| st.dynamic.remove(k)).collect();
        let slot = ((TP_TOP - tp) / TP_STEP) as usize;
        if let Some(s) = st.slots.get_mut(slot) {
            *s = false;
        }
        blocks
    };
    GENERATION.fetch_add(1, Ordering::Release);
    for (addr, size) in blocks {
        unmap(addr, size);
    }
    unmap(tp - STATIC_RESERVE, STATIC_RESERVE + TCB_SIZE);
}

/// Block of module `id` for thread `tp`: the static block, or the thread's
/// dynamic block (allocated and seeded on first use). Returns (address, size).
fn block(tp: u64, id: u64) -> Option<(u64, u64)> {
    let mut st = lock();
    let m = st.modules.iter().find(|m| m.id == id)?.clone();
    if let Some(off) = m.offset {
        return Some((tp - off, m.memsz));
    }
    if let Some(&b) = st.dynamic.get(&(tp, id)) {
        return Some((b.0, m.memsz));
    }
    // Dynamic block: zero-filled allocation seeded with the init image,
    // aligned as the segment requires.
    let size = (m.memsz + m.align.max(16) + 0xFFF) & !0xFFF;
    let base = map_rw(0, size, false)?;
    let addr = (base + m.align.max(1) - 1) & !(m.align.max(1) - 1);
    if !m.init.is_empty() {
        unsafe { std::ptr::copy_nonoverlapping(m.init.as_ptr(), addr as *mut u8, m.init.len()) };
    }
    st.dynamic.insert((tp, id), (base, size));
    if tp != 0 && ps5vm::ranges::accessible(tp + DTV_OFFSET, 8, true) {
        let dtv = tp + DTV_OFFSET;
        unsafe {
            std::ptr::write((dtv + 8 + id * 8) as *mut u64, addr);
            let count = (dtv + 8) as *mut u64;
            *count = (*count).max(id);
        }
    }
    Some((addr, m.memsz))
}

/// Address of `offset` within module `id`'s block for thread `tp`.
pub fn resolve(tp: u64, id: u64, offset: u64) -> Option<u64> {
    let generation = GENERATION.load(Ordering::Acquire);
    let cached = CACHE.with(|c| {
        let c = c.borrow();
        if c.0 != generation || c.1 != tp {
            return None;
        }
        c.2.iter().find(|e| e.0 == id).map(|e| (e.1, e.2))
    });
    let (addr, size) = match cached {
        Some(b) => b,
        None => {
            let b = block(tp, id)?;
            CACHE.with(|c| {
                let mut c = c.borrow_mut();
                if c.0 != generation || c.1 != tp {
                    *c = (generation, tp, Vec::new());
                }
                c.2.push((id, b.0, b.1));
            });
            b
        }
    };
    (offset < size).then(|| addr + offset)
}

/// __tls_get_addr(tls_index*) where tls_index = { u64 module, u64 offset }.
fn cb_tls_get_addr(ctx: &mut CpuContext) -> HleResult {
    let p = ctx.rdi();
    let (Some(id), Some(off)) = (ctx.read_u64(p), ctx.read_u64(p + 8)) else { return 0 };
    let tp = ps5cpu::exec::current_thread_pointer();
    resolve(tp, id, off).unwrap_or(0) as i64
}

pub fn register(reg: &mut HleRegistry) {
    reg.register_fn("vNe1w4diLCs", "__tls_get_addr", "libKernel", cb_tls_get_addr);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pthread object of a new thread block starts with the 64-bit
    /// thread id.
    #[test]
    fn pthread_object_starts_with_the_thread_id() {
        let vm = ps5vm::VirtualMemory::new();
        let tp = allocate_thread_block(&vm, 0xFFFF_FFF7).unwrap();
        let mut id = [0u8; 8];
        vm.read(tp + HANDLE_OFFSET, &mut id).unwrap();
        assert_eq!(u64::from_le_bytes(id), 0xFFFF_FFF7);
        free_thread_block(tp);
    }

    #[test]
    fn variant_two_offsets() {
        // eboot: memsz 0x18, align 8 ; libc: memsz 0x468, align 0x10
        let a = static_offset(0, 0x18, 8, 0);
        assert_eq!(a, 0x18);
        let b = static_offset(a, 0x468, 0x10, 0);
        assert_eq!(b % 0x10, 0);
        assert!(b >= a + 0x468);
    }

    /// A module registered after the first thread uses dynamic TLS: each
    /// thread gets its own lazily allocated block, seeded with the init
    /// image snapshot, and the blocks go away with the thread and module.
    #[test]
    fn dynamic_module_blocks() {
        const BASE: u64 = 0x5CA8_0000_0000;
        let vm = ps5vm::VirtualMemory::new();
        let tdata = BASE;
        vm.map(tdata, 0x1000, ps5vm::MemProtect::RW).unwrap();
        vm.write(tdata, &[1, 2, 3, 4]).unwrap();
        let tp1 = allocate_thread_block(&vm, 1).unwrap();
        let tp2 = allocate_thread_block(&vm, 2).unwrap();
        let (id, off) = register_module(16, 0x40, 4, tdata, 0).unwrap();
        assert_eq!(off, 0);
        assert!(is_dynamic(id));
        set_init_image(&vm, id);
        // The template changes after the snapshot: new blocks keep the snapshot.
        vm.write(tdata, &[9, 9, 9, 9]).unwrap();
        let a1 = resolve(tp1, id, 0).unwrap();
        let a2 = resolve(tp2, id, 0).unwrap();
        assert_ne!(a1, a2);
        assert_eq!(a1 % 16, 0);
        for a in [a1, a2] {
            assert_eq!(unsafe { std::slice::from_raw_parts(a as *const u8, 8) }, &[1, 2, 3, 4, 0, 0, 0, 0]);
        }
        // Per-thread state survives, and the DTV points at the block.
        unsafe { *(a1 as *mut u8) = 7 };
        assert_eq!(resolve(tp1, id, 0), Some(a1));
        assert_eq!(unsafe { *(a1 as *const u8) }, 7);
        assert_eq!(unsafe { *((tp1 + DTV_OFFSET + 8 + id * 8) as *const u64) }, a1);
        assert_eq!(resolve(tp1, id, 0x3F), Some(a1 + 0x3F));
        assert_eq!(resolve(tp1, id, 0x40), None);
        free_thread_block(tp2);
        assert!(!ps5vm::ranges::covered(a2, 1));
        unregister_module(id);
        assert!(!ps5vm::ranges::covered(a1, 1));
        assert_eq!(resolve(tp1, id, 0), None);
        free_thread_block(tp1);
    }
}
