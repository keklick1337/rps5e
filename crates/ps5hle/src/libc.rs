// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libc HLE implementations: malloc, free, memcpy, memset, strlen, etc.
//!
//! Guest addresses are identity-mapped onto host pages, so these operate
//! directly on guest virtual memory through the CpuContext.

#[cfg(windows)]
#[allow(unused_imports)]
use crate::host::c as libc;
use crate::context::{CpuContext, CpuRegister};
use crate::registry::{HleCallback, HleRegistry, HleResult};
use std::sync::Mutex;


/// OK result code (0 = success).
const OK: HleResult = 0;

// ---------------------------------------------------------------------------
// Guest heap allocator. A simple bump+free-list allocator over a dedicated
// guest memory region. PS5 malloc returns guest-visible addresses.
// ---------------------------------------------------------------------------

/// Base of the guest heap region (lives below the main image).
#[cfg(unix)]
const HEAP_BASE: u64 = 0x0000_0000_0100_0000;
/// Windows keeps its own DLLs and heaps in the low gigabytes.
#[cfg(windows)]
const HEAP_BASE: u64 = 0x0000_000E_0000_0000;
/// Initial heap size (16 MiB, grows on demand).
const HEAP_INITIAL_SIZE: u64 = 0x0100_0000;
/// Maximum heap size (256 MiB).
const HEAP_MAX_SIZE: u64 = 0x1000_0000;
/// Minimum allocation alignment. The console's compiler assumes 32 bytes
/// for a fresh malloc / operator new result (its default new alignment), so
/// guest code may store to it with 32-byte-aligned AVX instructions.
const DEFAULT_ALIGN: u64 = 32;

struct HeapBlock {
    addr: u64,
    size: u64,
    free: bool,
}

struct GuestHeap {
    blocks: Vec<HeapBlock>,
    committed: u64,
    initialized: bool,
}

static HEAP: Mutex<GuestHeap> = Mutex::new(GuestHeap {
    blocks: Vec::new(),
    committed: 0,
    initialized: false,
});

fn ensure_heap_init(vm: &VirtualMemory) {
    let mut heap = HEAP.lock().unwrap();
    if heap.initialized {
        return;
    }
    heap.initialized = true;
    heap.blocks.push(HeapBlock {
        addr: HEAP_BASE,
        size: HEAP_INITIAL_SIZE,
        free: true,
    });
    heap.committed = HEAP_INITIAL_SIZE;
    // Map the initial heap region as read-write.
    let _ = vm.map(HEAP_BASE, HEAP_INITIAL_SIZE, ps5vm::MemProtect::RW);
    log::info!(
        "[heap] initialized guest heap at {:#x} size {:#x}",
        HEAP_BASE,
        HEAP_INITIAL_SIZE
    );
}

use ps5vm::VirtualMemory;

fn align_up(v: u64, align: u64) -> u64 {
    (v + align - 1) & !(align - 1)
}

/// Allocate `size` bytes from the guest heap at a multiple of `alignment`
/// (a power of two; at least `DEFAULT_ALIGN` is used). Every block starts
/// and ends on a `DEFAULT_ALIGN` boundary.
fn heap_alloc(ctx: &CpuContext, size: u64, alignment: u64, zero: bool) -> u64 {
    if size == 0 {
        return 0;
    }
    let alignment = alignment.max(DEFAULT_ALIGN);
    // Larger than the whole heap (this also keeps the rounding below from
    // overflowing for sizes near 2^64).
    if size > HEAP_MAX_SIZE || alignment > HEAP_MAX_SIZE {
        log::warn!("[heap] refused allocation of {size:#x} bytes (alignment {alignment:#x})");
        return 0;
    }
    ensure_heap_init(ctx.vm);
    let mut heap = HEAP.lock().unwrap();
    let alloc_size = align_up(size, DEFAULT_ALIGN);

    // First-fit search through free blocks. Find the index first, then
    // mutate outside the borrow loop.
    let mut found_idx: Option<usize> = None;
    let mut split_lead: Option<(u64, u64)> = None; // (addr, size)
    let mut split_remaining: Option<(u64, u64)> = None; // (addr, size)
    for (i, block) in heap.blocks.iter_mut().enumerate() {
        if !block.free {
            continue;
        }
        let start = align_up(block.addr, alignment);
        let lead = start - block.addr;
        if block.size < lead || block.size - lead < alloc_size {
            continue;
        }
        let remaining = block.size - lead - alloc_size;
        if lead != 0 {
            split_lead = Some((block.addr, lead));
        }
        if remaining >= DEFAULT_ALIGN {
            split_remaining = Some((start + alloc_size, remaining));
            block.size = alloc_size;
        } else {
            block.size -= lead;
        }
        block.addr = start;
        block.free = false;
        found_idx = Some(i);
        break;
    }

    if let Some(idx) = found_idx {
        let addr = heap.blocks[idx].addr;
        for (new_addr, free_size) in [split_lead, split_remaining].into_iter().flatten() {
            heap.blocks.push(HeapBlock {
                addr: new_addr,
                size: free_size,
                free: true,
            });
        }
        drop(heap);
        if zero {
            let _ = ctx.vm.write(addr, &vec![0u8; alloc_size as usize]);
        }
        return addr;
    }

    // No free block fits: merge freed neighbours and look again before
    // growing the heap (up to HEAP_MAX_SIZE).
    if heap_coalesce(&mut heap) {
        drop(heap);
        return heap_alloc(ctx, size, alignment, zero);
    }
    let current_committed = heap.committed;
    let padding = if alignment > DEFAULT_ALIGN { alignment } else { 0 };
    let new_committed = (current_committed + alloc_size + padding + 0x10000).min(HEAP_MAX_SIZE);
    drop(heap);
    if new_committed > current_committed {
        let _ = ctx.vm.map(
            HEAP_BASE + current_committed,
            new_committed - current_committed,
            ps5vm::MemProtect::RW,
        );
        {
            let mut heap = HEAP.lock().unwrap();
            heap.committed = new_committed;
            heap.blocks.push(HeapBlock {
                addr: HEAP_BASE + current_committed,
                size: new_committed - current_committed,
                free: true,
            });
        }
        return heap_alloc(ctx, size, alignment, zero);
    }

    log::warn!("[heap] out of memory: requested {:#x}", size);
    0
}

/// Free a previously allocated block.
fn heap_free(addr: u64) {
    if addr == 0 {
        return;
    }
    let mut heap = HEAP.lock().unwrap();
    for block in &mut heap.blocks {
        if block.addr == addr && !block.free {
            block.free = true;
            return;
        }
    }
    // Freeing an unknown address — ignore silently (guest may double-free).
}

/// Size of the allocated block at `addr`.
fn heap_block_size(addr: u64) -> Option<u64> {
    let heap = HEAP.lock().unwrap();
    heap.blocks.iter().find(|b| b.addr == addr && !b.free).map(|b| b.size)
}

/// Merge adjacent free blocks (the list is kept unsorted between calls);
/// true when anything merged.
fn heap_coalesce(heap: &mut GuestHeap) -> bool {
    let before = heap.blocks.len();
    heap.blocks.sort_unstable_by_key(|b| b.addr);
    let mut out: Vec<HeapBlock> = Vec::with_capacity(before);
    for b in heap.blocks.drain(..) {
        match out.last_mut() {
            Some(last) if last.free && b.free && last.addr + last.size == b.addr => last.size += b.size,
            _ => out.push(b),
        }
    }
    heap.blocks = out;
    heap.blocks.len() != before
}

// ---------------------------------------------------------------------------
// HLE callback implementations
// ---------------------------------------------------------------------------

const ENOMEM: i32 = 12;

/// An allocation of `size` bytes (a request for 0 bytes still returns a
/// unique block, as the console's allocator does); a failure sets ENOMEM.
fn guest_alloc(ctx: &CpuContext, size: u64, alignment: u64, zero: bool) -> u64 {
    let addr = heap_alloc(ctx, size.max(1), alignment, zero);
    if addr == 0 {
        crate::kernel::set_errno(ctx, ENOMEM);
    }
    addr
}

/// A block of the HLE heap for libc functions that return malloc'd
/// memory (0 with ENOMEM on failure).
pub(crate) fn heap_block(ctx: &CpuContext, size: u64) -> u64 {
    guest_alloc(ctx, size, DEFAULT_ALIGN, false)
}

/// Free a block from [`heap_block`] (NULL is ignored).
pub(crate) fn heap_release(addr: u64) {
    heap_free(addr);
}

fn cb_malloc(ctx: &mut CpuContext) -> HleResult {
    guest_alloc(ctx, ctx.rdi(), DEFAULT_ALIGN, false) as i64
}

fn cb_calloc(ctx: &mut CpuContext) -> HleResult {
    let (count, size) = (ctx.rdi(), ctx.rsi());
    let Some(total) = count.checked_mul(size) else {
        crate::kernel::set_errno(ctx, ENOMEM);
        return 0;
    };
    guest_alloc(ctx, total, DEFAULT_ALIGN, true) as i64
}

/// memalign(alignment, size): the alignment must be a power of two.
fn cb_memalign(ctx: &mut CpuContext) -> HleResult {
    let (alignment, size) = (ctx.rdi(), ctx.rsi());
    if !alignment.is_power_of_two() {
        crate::kernel::set_errno(ctx, EINVAL);
        return 0;
    }
    guest_alloc(ctx, size, alignment, false) as i64
}

/// aligned_alloc(alignment, size): as memalign.
fn cb_aligned_alloc(ctx: &mut CpuContext) -> HleResult {
    cb_memalign(ctx)
}

/// posix_memalign(void** result, alignment, size) -> 0 / EINVAL / ENOMEM.
/// The alignment must be a power of two and a multiple of a pointer; on
/// failure `*result` is left alone and errno is not touched.
fn cb_posix_memalign(ctx: &mut CpuContext) -> HleResult {
    let (out, alignment, size) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    if out == 0 || !alignment.is_power_of_two() || alignment < 8 {
        return EINVAL as HleResult;
    }
    let addr = heap_alloc(ctx, size.max(1), alignment, false);
    if addr == 0 {
        return ENOMEM as HleResult;
    }
    ctx.write_u64(out, addr);
    OK
}

fn cb_realloc(ctx: &mut CpuContext) -> HleResult {
    let old_addr = ctx.rdi();
    let new_size = ctx.rsi();
    if old_addr == 0 {
        return guest_alloc(ctx, new_size, DEFAULT_ALIGN, false) as i64;
    }
    if new_size == 0 {
        heap_free(old_addr);
        return 0;
    }
    // The old block's size bounds the copy: reading new_size bytes of a
    // smaller block reads past it (and nothing at all when the tail is
    // unmapped).
    let Some(old_size) = heap_block_size(old_addr) else {
        log::warn!("[heap] realloc of {old_addr:#x}, which is not an allocated block");
        return 0;
    };
    let new_addr = guest_alloc(ctx, new_size, DEFAULT_ALIGN, false);
    if new_addr != 0 {
        crate::memops::copy("realloc", new_addr, old_addr, old_size.min(new_size), crate::memops::caller(ctx));
        heap_free(old_addr);
    }
    new_addr as i64
}

/// malloc_usable_size(ptr): the size of the block, at least what was
/// requested (the heap rounds blocks up to its alignment); 0 for NULL and
/// for an address that is not an allocated block.
fn cb_malloc_usable_size(ctx: &mut CpuContext) -> HleResult {
    let addr = ctx.rdi();
    if addr == 0 {
        return 0;
    }
    match heap_block_size(addr) {
        Some(size) => size as HleResult,
        None => {
            log::warn!("[heap] malloc_usable_size of {addr:#x}, which is not an allocated block");
            0
        }
    }
}

fn cb_free(ctx: &mut CpuContext) -> HleResult {
    heap_free(ctx.rdi());
    OK
}

fn cb_memcpy(ctx: &mut CpuContext) -> HleResult {
    let (dst, src, len) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    crate::memops::copy("memcpy", dst, src, len, crate::memops::caller(ctx));
    // memcpy returns the destination pointer.
    dst as i64
}

fn cb_memmove(ctx: &mut CpuContext) -> HleResult {
    let (dst, src, len) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    crate::memops::copy("memmove", dst, src, len, crate::memops::caller(ctx));
    dst as i64
}

fn cb_memset(ctx: &mut CpuContext) -> HleResult {
    let (dst, val, len) = (ctx.rdi(), (ctx.rsi() & 0xFF) as u8, ctx.rdx());
    crate::memops::fill("memset", dst, val, len, crate::memops::caller(ctx));
    // memset returns the destination pointer.
    dst as i64
}

fn cb_memcmp(ctx: &mut CpuContext) -> HleResult {
    let a = ctx.rdi();
    let b = ctx.rsi();
    let len = ctx.rdx() as usize;
    if len == 0 {
        return 0;
    }
    let data_a = ctx.read_bytes(a, len);
    let data_b = ctx.read_bytes(b, len);
    match (data_a, data_b) {
        // Fast path: both ranges readable — a plain byte compare (a null
        // pointer reads as an empty buffer and takes the path below).
        (Some(da), Some(db)) if da.len() == len && db.len() == len => {
            for i in 0..len {
                if da[i] != db[i] {
                    return (da[i] as i64) - (db[i] as i64);
                }
            }
            0
        }
        // A range the registry refuses: compare what is readable and treat
        // an unreadable byte as 0 (the same zero-fill the other ops use),
        // so a partially mapped compare still orders correctly.
        _ => {
            let rd = |base: u64, i: usize| {
                ctx.read_bytes(base + i as u64, 1).and_then(|v| v.first().copied()).unwrap_or(0)
            };
            for i in 0..len {
                let (ba, bb) = (rd(a, i), rd(b, i));
                if ba != bb {
                    return (ba as i64) - (bb as i64);
                }
            }
            0
        }
    }
}

fn cb_strlen(ctx: &mut CpuContext) -> HleResult {
    ctx.read_cstr_bytes(ctx.rdi(), usize::MAX).map_or(0, |b| b.len() as i64)
}

fn cb_strnlen(ctx: &mut CpuContext) -> HleResult {
    let maxlen = usize::try_from(ctx.rsi()).unwrap_or(usize::MAX);
    ctx.read_cstr_bytes(ctx.rdi(), maxlen).map_or(0, |b| b.len() as i64)
}

/// Unsigned byte comparison of two C strings, looking at most at `n` bytes.
fn compare_cstr(ctx: &CpuContext, a: u64, b: u64, n: usize) -> HleResult {
    let a = ctx.read_cstr_bytes(a, n).unwrap_or_default();
    let b = ctx.read_cstr_bytes(b, n).unwrap_or_default();
    let common = a.len().min(b.len());
    if let Some(i) = (0..common).find(|&i| a[i] != b[i]) {
        return a[i] as i64 - b[i] as i64;
    }
    if common == n {
        return 0;
    }
    // One string ended: its terminator compares as 0.
    a.get(common).copied().unwrap_or(0) as i64 - b.get(common).copied().unwrap_or(0) as i64
}

fn cb_strcmp(ctx: &mut CpuContext) -> HleResult {
    compare_cstr(ctx, ctx.rdi(), ctx.rsi(), usize::MAX)
}

fn cb_strncmp(ctx: &mut CpuContext) -> HleResult {
    let n = usize::try_from(ctx.rdx()).unwrap_or(usize::MAX);
    if n == 0 {
        return 0;
    }
    compare_cstr(ctx, ctx.rdi(), ctx.rsi(), n)
}

fn cb_strcpy(ctx: &mut CpuContext) -> HleResult {
    let dst = ctx.rdi();
    let src = ctx.rsi();
    match ctx.read_cstr_bytes(src, usize::MAX) {
        Some(mut bytes) => {
            bytes.push(0);
            crate::memops::store("strcpy", dst, &bytes, crate::memops::caller(ctx));
        }
        None => crate::memops::note_failure("strcpy", src, 1, false, crate::memops::caller(ctx)),
    }
    dst as i64
}

fn cb_stack_chk_fail(ctx: &mut CpuContext) -> HleResult {
    // The stack protector fired: the frame is corrupt, the process aborts.
    let caller = ctx.read_u64(ctx.get(crate::context::CpuRegister::Rsp)).unwrap_or(0);
    log::error!("[libc] __stack_chk_fail called from {caller:#x}; aborting");
    ps5vm::shutdown::exit(134)
}

fn cb_vsnprintf(ctx: &mut CpuContext) -> HleResult {
    // vsnprintf(dst, size, format, va_list)
    let (dst, size, fmt_addr, va_list) = (ctx.rdi(), ctx.rsi(), ctx.rdx(), ctx.rcx());
    let format = ctx.read_cstr_bytes(fmt_addr, MAX_FORMAT).unwrap_or_default();
    let rendered = match crate::printf::VaListCursor::new(ctx, va_list) {
        Some(mut cursor) => crate::printf::format_bytes(ctx, &format, &mut cursor),
        None => format,
    };
    write_c_output(ctx, dst, size, &rendered)
}

/// Longest input string the sscanf-style fallbacks read.
const MAX_SCAN_INPUT: usize = 1 << 20;

/// Scan `input` units with `format` under the `_s` rules.
fn scan_secure(ctx: &CpuContext, format: &[u16], input: &[u16], args: &mut dyn crate::printf::ArgSource, wide_input: bool) -> HleResult {
    let mut pos = 0usize;
    let mut src = || {
        let c = input.get(pos).copied();
        if c.is_some() {
            pos += 1;
        }
        c
    };
    let opts = crate::scanf::Options { secure: true, wide_input };
    crate::scanf::scan_units(ctx, format, &mut src, args, opts).0
}

/// vsscanf_s(const char* s, const char* format, va_list): sscanf with a
/// capacity after every c, s and [ target; a null string, format or
/// target is EOF.
fn cb_vsscanf_s(ctx: &mut CpuContext) -> HleResult {
    let (s, fmt_addr, va_list) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    if s == 0 || fmt_addr == 0 {
        return crate::scanf::EOF;
    }
    let widen = |b: Vec<u8>| -> Vec<u16> { b.into_iter().map(u16::from).collect() };
    let input = widen(ctx.read_cstr_bytes(s, MAX_SCAN_INPUT).unwrap_or_default());
    let format = widen(ctx.read_cstr_bytes(fmt_addr, MAX_FORMAT).unwrap_or_default());
    let Some(mut cursor) = crate::printf::VaListCursor::new(ctx, va_list) else { return crate::scanf::EOF };
    scan_secure(ctx, &format, &input, &mut cursor, false)
}

/// swscanf_s(const wchar_t* s, const wchar_t* format, ...): the wide
/// vsscanf_s over 16-bit guest units.
fn cb_swscanf_s(ctx: &mut CpuContext) -> HleResult {
    let (s, fmt_addr) = (ctx.rdi(), ctx.rsi());
    if s == 0 || fmt_addr == 0 {
        return crate::scanf::EOF;
    }
    let input = ctx.read_wide_string(s, MAX_SCAN_INPUT).unwrap_or_default();
    let format = ctx.read_wide_string(fmt_addr, MAX_FORMAT).unwrap_or_default();
    let mut args = crate::printf::RegArgsCursor::new(ctx, 2);
    scan_secure(ctx, &format, &input, &mut args, true)
}

/// vsprintf(dst, format, va_list): vsnprintf without a size bound.
fn cb_vsprintf(ctx: &mut CpuContext) -> HleResult {
    let (dst, fmt_addr, va_list) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    let format = ctx.read_cstr_bytes(fmt_addr, MAX_FORMAT).unwrap_or_default();
    let rendered = match crate::printf::VaListCursor::new(ctx, va_list) {
        Some(mut cursor) => crate::printf::format_bytes(ctx, &format, &mut cursor),
        None => format,
    };
    write_c_output(ctx, dst, u64::MAX, &rendered)
}

/// snprintf(dst, size, format, ...)
fn cb_snprintf(ctx: &mut CpuContext) -> HleResult {
    let (dst, size) = (ctx.rdi(), ctx.rsi());
    let format = ctx.read_cstr_bytes(ctx.rdx(), MAX_FORMAT).unwrap_or_default();
    let rendered = {
        let mut args = crate::printf::RegArgsCursor::new(ctx, 3);
        crate::printf::format_bytes(ctx, &format, &mut args)
    };
    write_c_output(ctx, dst, size, &rendered)
}

/// Longest format string read from the guest.
const MAX_FORMAT: usize = 1 << 20;

/// Send printf output to the log; returns the produced byte count.
fn print_rendered(rendered: &[u8]) -> HleResult {
    let visible = rendered.split(|&b| b == 0).next().unwrap_or_default();
    if !visible.is_empty() {
        log::info!("[stdout] {}", String::from_utf8_lossy(visible).trim_end_matches('\n'));
    }
    rendered.len() as i64
}

/// vprintf(format, va_list)
fn cb_vprintf(ctx: &mut CpuContext) -> HleResult {
    let format = ctx.read_cstr_bytes(ctx.rdi(), MAX_FORMAT).unwrap_or_default();
    let rendered = match crate::printf::VaListCursor::new(ctx, ctx.rsi()) {
        Some(mut cursor) => crate::printf::format_bytes(ctx, &format, &mut cursor),
        None => format,
    };
    print_rendered(&rendered)
}

/// printf_s(format, ...) -> bytes / -1: printf under the `_s` rules (C11
/// K.3.5.3.3): a null format, any %n and a null string argument return -1
/// and print nothing.
fn cb_printf_s(ctx: &mut CpuContext) -> HleResult {
    let fmt_addr = ctx.rdi();
    let mut args = crate::printf::RegArgsCursor::new(ctx, 1);
    printf_s_core(ctx, fmt_addr, Some(&mut args))
}

/// vprintf_s(format, va_list): [`cb_printf_s`] over a va_list.
fn cb_vprintf_s(ctx: &mut CpuContext) -> HleResult {
    let (fmt_addr, va_list) = (ctx.rdi(), ctx.rsi());
    let mut cursor = crate::printf::VaListCursor::new(ctx, va_list);
    printf_s_core(ctx, fmt_addr, cursor.as_mut().map(|c| c as &mut dyn crate::printf::ArgSource))
}

fn printf_s_core(ctx: &CpuContext, fmt_addr: u64, args: Option<&mut dyn crate::printf::ArgSource>) -> HleResult {
    if fmt_addr == 0 {
        return -1;
    }
    let (Some(format), Some(args)) = (ctx.read_cstr_bytes(fmt_addr, MAX_FORMAT), args) else { return -1 };
    match crate::printf::format_bytes_secure(ctx, &format, args) {
        Some(rendered) => print_rendered(&rendered),
        None => -1,
    }
}

/// Write rendered bytes into a bounded guest buffer (snprintf semantics:
/// at most size-1 bytes plus the terminator). Returns the full length.
fn write_c_output(ctx: &mut CpuContext, dst: u64, size: u64, rendered: &[u8]) -> HleResult {
    if dst != 0 && size != 0 {
        let n = rendered.len().min((size - 1) as usize);
        let mut bytes = Vec::with_capacity(n + 1);
        bytes.extend_from_slice(&rendered[..n]);
        bytes.push(0);
        ctx.write_bytes(dst, &bytes);
    }
    rendered.len() as i64
}

/// Longest wide format read from the guest, in units.
const MAX_WIDE_FORMAT: usize = 1 << 16;

/// Store wide output into a bounded guest buffer (vswprintf semantics):
/// at most `size - 1` units and a terminator. Returns the unit count, or
/// -1 when the output did not fit (swprintf reports truncation as an error,
/// unlike snprintf); a rejected format leaves an empty string, EINVAL.
fn write_wide_output(ctx: &CpuContext, dst: u64, size_units: u64, rendered: Option<Vec<u16>>) -> HleResult {
    let Some(units) = rendered else {
        ctx.write_bytes(dst, b"\0\0");
        crate::kernel::set_errno(ctx, EINVAL);
        return -1;
    };
    let n = units.len().min((size_units - 1).min(usize::MAX as u64) as usize);
    let mut bytes: Vec<u8> = units[..n].iter().flat_map(|u| u.to_le_bytes()).collect();
    bytes.extend_from_slice(b"\0\0");
    ctx.write_bytes(dst, &bytes);
    if n == units.len() { n as HleResult } else { -1 }
}

/// Shared body of the swprintf family: `args` supplies the variadic part.
/// A null buffer or a zero size is EINVAL and nothing is written.
fn swprintf_core(ctx: &CpuContext, dst: u64, size: u64, fmt_addr: u64, args: Option<&mut dyn crate::printf::ArgSource>, secure: bool) -> HleResult {
    if dst == 0 || size == 0 {
        crate::kernel::set_errno(ctx, EINVAL);
        return -1;
    }
    let rendered = match (ctx.read_wide_string(fmt_addr, MAX_WIDE_FORMAT), args) {
        (Some(format), Some(args)) => crate::printf::format_wide(ctx, &format, args, secure),
        _ => None,
    };
    write_wide_output(ctx, dst, size, rendered)
}

/// vswprintf(dst, size, wide_format, va_list)
fn cb_vswprintf(ctx: &mut CpuContext) -> HleResult {
    let (dst, size, fmt_addr, va_list) = (ctx.rdi(), ctx.rsi(), ctx.rdx(), ctx.rcx());
    let mut cursor = crate::printf::VaListCursor::new(ctx, va_list);
    let args = cursor.as_mut().map(|c| c as &mut dyn crate::printf::ArgSource);
    swprintf_core(ctx, dst, size, fmt_addr, args, false)
}

/// vswprintf_s(dst, size, wide_format, va_list): %n and null strings are
/// constraint violations.
fn cb_vswprintf_s(ctx: &mut CpuContext) -> HleResult {
    let (dst, size, fmt_addr, va_list) = (ctx.rdi(), ctx.rsi(), ctx.rdx(), ctx.rcx());
    let mut cursor = crate::printf::VaListCursor::new(ctx, va_list);
    let args = cursor.as_mut().map(|c| c as &mut dyn crate::printf::ArgSource);
    swprintf_core(ctx, dst, size, fmt_addr, args, true)
}

/// swprintf(dst, size, wide_format, ...) — variadic via registers/stack.
fn cb_swprintf(ctx: &mut CpuContext) -> HleResult {
    let (dst, size, fmt_addr) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    let mut args = crate::printf::RegArgsCursor::new(ctx, 3);
    swprintf_core(ctx, dst, size, fmt_addr, Some(&mut args), false)
}

/// swprintf_s(dst, size, wide_format, ...): as swprintf, %n and null
/// strings are constraint violations.
fn cb_swprintf_s(ctx: &mut CpuContext) -> HleResult {
    let (dst, size, fmt_addr) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    let mut args = crate::printf::RegArgsCursor::new(ctx, 3);
    swprintf_core(ctx, dst, size, fmt_addr, Some(&mut args), true)
}

/// wcslen(wide_str) — number of wide chars before the NUL terminator.
fn cb_wcslen(ctx: &mut CpuContext) -> HleResult {
    let addr = ctx.rdi();
    if addr == 0 {
        return 0;
    }
    let units = ctx.read_wide_string(addr, 1_048_576).unwrap_or_default();
    units.len() as i64
}

fn cb_printf(ctx: &mut CpuContext) -> HleResult {
    // printf(format, ...): rendered into the log.
    let format = ctx.read_cstr_bytes(ctx.rdi(), MAX_FORMAT).unwrap_or_default();
    let rendered = {
        let mut c = crate::printf::RegArgsCursor::new(ctx, 1);
        crate::printf::format_bytes(ctx, &format, &mut c)
    };
    print_rendered(&rendered)
}

/// puts(s): the string and a newline; returns the bytes written.
fn cb_puts(ctx: &mut CpuContext) -> HleResult {
    let mut text = ctx.read_cstr_bytes(ctx.rdi(), MAX_FORMAT).unwrap_or_else(|| b"(null)".to_vec());
    text.push(b'\n');
    print_rendered(&text)
}

/// abort(): abnormal process termination (SIGABRT status).
fn cb_abort(ctx: &mut CpuContext) -> HleResult {
    let caller = ctx.read_u64(ctx.get(CpuRegister::Rsp)).unwrap_or(0);
    log::error!("[libc] abort() called from {caller:#x}");
    // Dump the code around the call site and any nearby assert string so the
    // reason can be identified offline (the guest is about to die).
    let start = caller.saturating_sub(0x60);
    let mut code = [0u8; 0xa0];
    let _ = ctx.vm.read(start, &mut code);
    log::error!("[libc] abort site {:#x} bytes: {:02x?}", start, &code);
    // Find `lea rax,[rip+disp32]` (48 8d 05 xx xx xx xx) in the abort path
    // (the bytes just before the ud2 after the caller) and print the string
    // it points to — that is the assert message.
    for i in 0..code.len().saturating_sub(7) {
        if code[i] == 0x48 && code[i + 1] == 0x8d && code[i + 2] == 0x05 {
            let disp = i32::from_le_bytes([
                code[i + 3], code[i + 4], code[i + 5], code[i + 6],
            ]) as i64;
            let saddr = (start + i as u64 + 7) as i64 + disp;
            if saddr > 0 {
                if let Some(s) = ctx.read_cstring(saddr as u64, 256) {
                    if !s.is_empty() && s.chars().any(|c| c.is_ascii_graphic()) {
                        log::error!("[libc] abort message @ {saddr:#x}: {s:?}");
                    }
                }
            }
        }
    }
    // The abort is usually reached from a failed assert; the message pointer
    // is often in rdi (or a lea near the call site). Try rdi/rsi/rdx/rcx/r8/r9
    // as C strings.
    for (name, v) in [
        ("rdi", ctx.rdi()),
        ("rsi", ctx.rsi()),
        ("rdx", ctx.rdx()),
        ("rcx", ctx.rcx()),
        ("r8", ctx.r8()),
        ("r9", ctx.r9()),
    ] {
        if let Some(s) = ctx.read_cstring(v, 256) {
            if !s.is_empty() && s.chars().any(|c| c.is_ascii_graphic()) {
                log::error!("[libc] abort {name} -> {s:?}");
            }
        }
    }
    ps5vm::shutdown::exit(134)
}

/// exit(status) when the game ships no libc: the calling thread's
/// thread_local destructors, then every atexit / __cxa_atexit handler,
/// latest first, while the emulator is still fully alive; only then the
/// process ends.
fn cb_exit(ctx: &mut CpuContext) -> HleResult {
    let status = ctx.rdi() as i32;
    log::info!("[libc] exit({status})");
    run_thread_atexit_destructors();
    crate::cxx::run_exit_handlers(0);
    ps5vm::shutdown::exit(status)
}

// ---------------------------------------------------------------------------
// String functions (served when the game's libc module does not export them)
// ---------------------------------------------------------------------------

/// Longest C string these helpers read from the guest.
const MAX_CSTR: usize = 1 << 24;

fn cstr(ctx: &CpuContext, addr: u64) -> Vec<u8> {
    ctx.read_cstr_bytes(addr, MAX_CSTR).unwrap_or_default()
}

const ERANGE: i32 = 34;
const EINVAL: i32 = 22;

/// strcpy_s(dest, destsz, src) -> errno_t. On failure dest becomes "".
fn cb_strcpy_s(ctx: &mut CpuContext) -> HleResult {
    let (dst, size, src) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    if dst == 0 || size == 0 {
        return EINVAL as HleResult;
    }
    if src == 0 {
        ctx.write_bytes(dst, &[0]);
        return EINVAL as HleResult;
    }
    let mut bytes = cstr(ctx, src);
    if bytes.len() as u64 >= size {
        ctx.write_bytes(dst, &[0]);
        return ERANGE as HleResult;
    }
    bytes.push(0);
    crate::memops::store("strcpy_s", dst, &bytes, crate::memops::caller(ctx));
    OK
}

/// strncpy(dest, src, n): copies at most n bytes, zero-filling the rest.
fn cb_strncpy(ctx: &mut CpuContext) -> HleResult {
    let (dst, src, n) = (ctx.rdi(), ctx.rsi(), ctx.rdx() as usize);
    if n != 0 {
        let caller = crate::memops::caller(ctx);
        let mut bytes = ctx.read_cstr_bytes(src, n).unwrap_or_else(|| {
            crate::memops::note_failure("strncpy", src, 1, false, caller);
            Vec::new()
        });
        bytes.resize(n, 0);
        crate::memops::store("strncpy", dst, &bytes, caller);
    }
    dst as HleResult
}

/// strcat(dest, src)
fn cb_strcat(ctx: &mut CpuContext) -> HleResult {
    let (dst, src) = (ctx.rdi(), ctx.rsi());
    let end = dst + cstr(ctx, dst).len() as u64;
    let mut bytes = cstr(ctx, src);
    bytes.push(0);
    crate::memops::store("strcat", end, &bytes, crate::memops::caller(ctx));
    dst as HleResult
}

/// strchr(s, c): the terminator itself can be found.
fn cb_strchr(ctx: &mut CpuContext) -> HleResult {
    let (s, c) = (ctx.rdi(), ctx.rsi() as u8);
    let bytes = cstr(ctx, s);
    if c == 0 {
        return (s + bytes.len() as u64) as HleResult;
    }
    bytes.iter().position(|&b| b == c).map_or(0, |i| (s + i as u64) as HleResult)
}

/// strrchr(s, c)
fn cb_strrchr(ctx: &mut CpuContext) -> HleResult {
    let (s, c) = (ctx.rdi(), ctx.rsi() as u8);
    let bytes = cstr(ctx, s);
    if c == 0 {
        return (s + bytes.len() as u64) as HleResult;
    }
    bytes.iter().rposition(|&b| b == c).map_or(0, |i| (s + i as u64) as HleResult)
}

/// strstr(haystack, needle): an empty needle matches at the start.
fn cb_strstr(ctx: &mut CpuContext) -> HleResult {
    let (h, n) = (ctx.rdi(), ctx.rsi());
    let hay = cstr(ctx, h);
    let needle = cstr(ctx, n);
    if needle.is_empty() {
        return h as HleResult;
    }
    hay.windows(needle.len()).position(|w| w == needle).map_or(0, |i| (h + i as u64) as HleResult)
}

/// Run a host strto* conversion over a guest string, storing *endptr.
fn host_strto<T>(ctx: &CpuContext, f: impl FnOnce(*const libc::c_char, *mut *mut libc::c_char) -> T) -> Option<T> {
    let (s, endptr) = (ctx.rdi(), ctx.rsi());
    if s == 0 {
        return None;
    }
    let text = std::ffi::CString::new(cstr(ctx, s)).ok()?;
    let mut end: *mut libc::c_char = std::ptr::null_mut();
    // The conversion reports overflow (ERANGE) through the host errno; the
    // guest checks its own (TCB +0x40).
    // SAFETY: the host thread's errno location.
    unsafe { *libc::__errno_location() = 0 };
    let value = f(text.as_ptr(), &mut end);
    // SAFETY: as above.
    let host_errno = unsafe { *libc::__errno_location() };
    if host_errno == libc::ERANGE {
        crate::kernel::set_errno(ctx, ERANGE);
    }
    if endptr != 0 {
        let consumed = (end as usize).wrapping_sub(text.as_ptr() as usize) as u64;
        ctx.write_u64(endptr, s + consumed);
    }
    Some(value)
}

/// strtol(str, endptr, base)
fn cb_strtol(ctx: &mut CpuContext) -> HleResult {
    let base = ctx.rdx() as i32;
    host_strto(ctx, |p, e| unsafe { libc::strtol(p, e, base) }).unwrap_or(0)
}

/// strtoul(str, endptr, base); also _Stoul.
fn cb_strtoul(ctx: &mut CpuContext) -> HleResult {
    let base = ctx.rdx() as i32;
    host_strto(ctx, |p, e| unsafe { libc::strtoul(p, e, base) }).unwrap_or(0) as HleResult
}

/// atoi(str)
fn cb_atoi(ctx: &mut CpuContext) -> HleResult {
    let s = ctx.rdi();
    let Ok(text) = std::ffi::CString::new(cstr(ctx, s)) else { return 0 };
    (unsafe { libc::atoi(text.as_ptr()) }) as HleResult
}

/// atol(str) / atoll(str): strtol in base 10 on the guest's 64-bit long;
/// the guest errno is not touched.
fn cb_atol(ctx: &mut CpuContext) -> HleResult {
    let s = ctx.rdi();
    let Ok(text) = std::ffi::CString::new(cstr(ctx, s)) else { return 0 };
    // SAFETY: a NUL-terminated host copy.
    (unsafe { libc::strtoll(text.as_ptr(), std::ptr::null_mut(), 10) }) as HleResult
}

/// tzset(): re-read the zone the time functions convert with (the host's;
/// the guest environment does not reach them).
fn cb_tzset(_ctx: &mut CpuContext) -> HleResult {
    // SAFETY: plain libc call.
    unsafe { tzset() };
    0
}

extern "C" {
    fn tzset();
}

/// The guest's host name, the same on every host: empty, as a process on
/// a PS5 reads it (firmware 13.60).
pub(crate) const GUEST_HOSTNAME: &[u8] = b"\0";
/// POSIX ENAMETOOLONG.
const ENAMETOOLONG: i32 = 63;

/// gethostname(name, len): copies at most `len` bytes of the name and its
/// terminator; a buffer without room for both is ENAMETOOLONG (the copied
/// prefix stays). A NULL name returns 0. errno is untouched on success.
fn cb_gethostname(ctx: &mut CpuContext) -> HleResult {
    let (name, len) = (ctx.rdi(), ctx.rsi());
    if name == 0 {
        return 0;
    }
    let count = (len as usize).min(GUEST_HOSTNAME.len());
    if count != 0 {
        ctx.write_bytes(name, &GUEST_HOSTNAME[..count]);
    }
    if count < GUEST_HOSTNAME.len() {
        crate::kernel::set_errno(ctx, ENAMETOOLONG);
        return -1;
    }
    0
}

// ---------------------------------------------------------------------------
// Time functions. The guest `struct tm` is FreeBSD's: nine 32-bit fields,
// then `long tm_gmtoff` at +40 and `const char* tm_zone` at +48 (56 bytes).
// ---------------------------------------------------------------------------

const GUEST_TM_SIZE: usize = 56;
const TM_GMTOFF: usize = 40;
const TM_ZONE: usize = 48;

/// Guest copies of the zone names `tm_zone` points to (one per name, kept
/// for the life of the process like the C library's own).
static ZONE_NAMES: Mutex<Vec<(Vec<u8>, u64)>> = Mutex::new(Vec::new());

/// Guest address of a NUL-terminated copy of `name` (0 when none fits).
fn guest_zone_name(ctx: &CpuContext, name: &[u8]) -> u64 {
    let mut names = ZONE_NAMES.lock().unwrap();
    if let Some(&(_, addr)) = names.iter().find(|(n, _)| n == name) {
        return addr;
    }
    let Some(addr) = crate::pthread::alloc_guest_block(ctx, name.len() as u64 + 1) else { return 0 };
    let mut bytes = name.to_vec();
    bytes.push(0);
    ctx.write_bytes(addr, &bytes);
    names.push((name.to_vec(), addr));
    addr
}

/// The guest layout of a host `tm`: the calendar fields, the UTC offset
/// and the guest address of the zone name.
fn tm_to_guest(t: &libc::tm, zone: u64) -> [u8; GUEST_TM_SIZE] {
    let fields = [t.tm_sec, t.tm_min, t.tm_hour, t.tm_mday, t.tm_mon, t.tm_year, t.tm_wday, t.tm_yday, t.tm_isdst];
    let mut out = [0u8; GUEST_TM_SIZE];
    for (i, f) in fields.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&f.to_le_bytes());
    }
    out[TM_GMTOFF..TM_GMTOFF + 8].copy_from_slice(&(t.tm_gmtoff as i64).to_le_bytes());
    out[TM_ZONE..TM_ZONE + 8].copy_from_slice(&zone.to_le_bytes());
    out
}

/// The guest `tm` of a host local-time result (its zone name copied to the
/// guest).
fn local_tm_to_guest(ctx: &CpuContext, t: &libc::tm) -> [u8; GUEST_TM_SIZE] {
    let zone = if t.tm_zone.is_null() {
        0
    } else {
        // SAFETY: the C library's zone names are static NUL-terminated strings.
        guest_zone_name(ctx, unsafe { std::ffi::CStr::from_ptr(t.tm_zone) }.to_bytes())
    };
    tm_to_guest(t, zone)
}

/// The calendar fields of a guest `tm` (the zone fields are not read: the
/// conversions recompute them).
fn tm_from_guest(ctx: &CpuContext, addr: u64) -> Option<libc::tm> {
    // The console layout: nine int fields (isdst at +32), padding, then
    // gmtoff at +40 and the zone name pointer at +48. The name is not
    // read: %Z formats the zone state, not the tm (measured on hardware).
    let b = ctx.read_bytes(addr, 48)?;
    let f = |i: usize| i32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap());
    let mut t: libc::tm = unsafe { std::mem::zeroed() };
    t.tm_sec = f(0);
    t.tm_min = f(1);
    t.tm_hour = f(2);
    t.tm_mday = f(3);
    t.tm_mon = f(4);
    t.tm_year = f(5);
    t.tm_wday = f(6);
    t.tm_yday = f(7);
    t.tm_isdst = f(8);
    t.tm_gmtoff = i64::from_le_bytes(b[40..48].try_into().unwrap()) as libc::c_long;
    Some(t)
}

/// UTC calendar fields of `time` from its day count (proleptic Gregorian),
/// for every time whose year fits an `int`; tm_gmtoff 0, no zone yet.
fn utc_calendar(time: i64) -> Option<libc::tm> {
    const SECONDS_PER_DAY: i64 = 86400;
    let days = time.div_euclid(SECONDS_PER_DAY);
    let second_of_day = time.rem_euclid(SECONDS_PER_DAY);
    let shifted = days + 719_468;
    let era = shifted.div_euclid(146_097);
    let day_of_era = shifted - era * 146_097;
    let year_of_era = (day_of_era - day_of_era / 1460 + day_of_era / 36524 - day_of_era / 146_096) / 365;
    let day_of_march_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let march_month = (5 * day_of_march_year + 2) / 153;
    let month = if march_month < 10 { march_month + 3 } else { march_month - 9 };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    let tm_year = i32::try_from(year - 1900).ok()?;
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let mut t: libc::tm = unsafe { std::mem::zeroed() };
    t.tm_sec = (second_of_day % 60) as i32;
    t.tm_min = (second_of_day / 60 % 60) as i32;
    t.tm_hour = (second_of_day / 3600) as i32;
    t.tm_mday = (day_of_march_year - (153 * march_month + 2) / 5 + 1) as i32;
    t.tm_mon = (month - 1) as i32;
    t.tm_year = tm_year;
    t.tm_wday = (days.rem_euclid(7) + 4).rem_euclid(7) as i32;
    t.tm_yday = (if month <= 2 { day_of_march_year - 306 } else { day_of_march_year + 59 + i64::from(leap) }) as i32;
    t.tm_isdst = 0;
    Some(t)
}

/// time(time_t* timer)
fn cb_time(ctx: &mut CpuContext) -> HleResult {
    let now = unsafe { libc::time(std::ptr::null_mut()) } as i64;
    if ctx.rdi() != 0 {
        ctx.write_u64(ctx.rdi(), now as u64);
    }
    now
}

thread_local! {
    /// Per-thread guest `struct tm` returned by gmtime/localtime.
    static TM_RESULT: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// The guest `tm` of `time`: UTC (zone "UTC") or local time.
fn guest_broken_down(ctx: &CpuContext, time: i64, local: bool) -> Option<[u8; GUEST_TM_SIZE]> {
    if !local {
        let t = utc_calendar(time)?;
        return Some(tm_to_guest(&t, guest_zone_name(ctx, b"UTC")));
    }
    let t = time as libc::time_t;
    let mut host: libc::tm = unsafe { std::mem::zeroed() };
    // SAFETY: both pointers are valid for the call.
    if unsafe { libc::localtime_r(&t, &mut host) }.is_null() {
        return None;
    }
    Some(local_tm_to_guest(ctx, &host))
}

fn broken_down_time(ctx: &CpuContext, local: bool) -> HleResult {
    let timer = ctx.rdi();
    if timer == 0 {
        return 0;
    }
    let Some(t) = ctx.read_u64(timer) else { return 0 };
    let Some(guest) = guest_broken_down(ctx, t as i64, local) else { return 0 };
    let mut out = TM_RESULT.with(|c| c.get());
    if out == 0 {
        let Some(p) = crate::pthread::alloc_guest_block(ctx, GUEST_TM_SIZE as u64) else { return 0 };
        TM_RESULT.with(|c| c.set(p));
        out = p;
    }
    if ctx.write_bytes(out, &guest) { out as HleResult } else { 0 }
}

/// gmtime_s(const time_t*, struct tm*) / localtime_s: into the caller's
/// tm, which is returned (NULL on failure).
fn broken_down_time_s(ctx: &CpuContext, local: bool) -> HleResult {
    let (timer, out) = (ctx.rdi(), ctx.rsi());
    if timer == 0 || out == 0 {
        return 0;
    }
    let Some(t) = ctx.read_u64(timer) else { return 0 };
    let Some(guest) = guest_broken_down(ctx, t as i64, local) else { return 0 };
    if ctx.write_bytes(out, &guest) { out as HleResult } else { 0 }
}

fn cb_gmtime_s(ctx: &mut CpuContext) -> HleResult {
    broken_down_time_s(ctx, false)
}

fn cb_localtime_s(ctx: &mut CpuContext) -> HleResult {
    broken_down_time_s(ctx, true)
}

/// difftime(time1, time0) -> double seconds.
fn cb_difftime(ctx: &mut CpuContext) -> HleResult {
    let (t1, t0) = (ctx.rdi() as i64, ctx.rsi() as i64);
    ctx.set_return_f64(t1 as f64 - t0 as f64);
    0
}

/// gmtime(const time_t*)
fn cb_gmtime(ctx: &mut CpuContext) -> HleResult {
    broken_down_time(ctx, false)
}

/// localtime(const time_t*)
fn cb_localtime(ctx: &mut CpuContext) -> HleResult {
    broken_down_time(ctx, true)
}

/// mktime(struct tm*): the local time of the fields, which are then
/// rewritten as localtime gives them (normalized, with the zone fields).
fn cb_mktime(ctx: &mut CpuContext) -> HleResult {
    let p = ctx.rdi();
    let Some(mut t) = (p != 0).then(|| tm_from_guest(ctx, p)).flatten() else { return -1 };
    // SAFETY: a valid tm.
    let result = unsafe { libc::mktime(&mut t) } as i64;
    if result == -1 {
        return -1;
    }
    if let Some(guest) = guest_broken_down(ctx, result, true) {
        ctx.write_bytes(p, &guest);
    }
    result
}

extern "C" {
    static tzname: [*const libc::c_char; 2];
}

/// The host's zone abbreviation for standard (false) or daylight time.
fn host_zone_name(dst: bool) -> Vec<u8> {
    // SAFETY: tzset initialises tzname, two valid C strings afterwards.
    unsafe {
        tzset();
        let p = tzname[usize::from(dst)];
        if p.is_null() { Vec::new() } else { std::ffi::CStr::from_ptr(p).to_bytes().to_vec() }
    }
}

// ---------------------------------------------------------------------------
// strftime / asctime format the conversions the console has, with the
// field and buffer behaviour measured on hardware.
// ---------------------------------------------------------------------------

const STRFTIME_WEEKDAYS: [&str; 7] = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const STRFTIME_WEEKDAYS_FULL: [&str; 7] = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
const STRFTIME_MONTHS: [&str; 12] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const STRFTIME_MONTHS_FULL: [&str; 12] = [
    "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December",
];

fn is_leap_year(year: i64) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

fn is_leap_year_sum(year: i64, base: i64) -> bool {
    is_leap_year(year % 400 + base % 400)
}

/// A weekday or month name: a negative index reads the first of the short
/// names, an index past the table the empty name.
fn table_name(index: i32, names: &[&'static str], short: &[&'static str]) -> &'static str {
    if index < 0 {
        short[0]
    } else if let Some(name) = names.get(index as usize) {
        name
    } else {
        ""
    }
}

/// A calendar field as the console prints it: a negative one reads 0, the
/// rest wrap at the field's modulus.
fn clamped_field(value: i64, modulus: i64) -> i64 {
    if value < 0 { 0 } else { value % modulus }
}

/// The strftime output cursor: pieces are dropped whole once one does not
/// fit the buffer.
struct FormatBuffer<'a> {
    out: &'a mut [u8],
    len: usize,
    /// A piece did not fit.
    full: bool,
    /// A conversion the console refuses happened (an impossible ISO week,
    /// a format ending after an `E` or `O`).
    bad: bool,
}

impl FormatBuffer<'_> {
    fn add_piece(&mut self, text: &[u8]) {
        if self.full {
            return;
        }
        if self.out.len() - self.len < text.len() {
            self.full = true;
            return;
        }
        self.out[self.len..self.len + text.len()].copy_from_slice(text);
        self.len += text.len();
    }

    fn add_text(&mut self, text: &str) {
        self.add_piece(text.as_bytes());
    }

    /// A number of at least `width` digits, `pad` filling a short one.
    fn add_number(&mut self, value: i64, modulus: i64, width: usize, pad: char) {
        let mut text = clamped_field(value, modulus).to_string();
        while text.len() < width {
            text.insert(0, pad);
        }
        self.add_text(&text);
    }
}

fn two_digit_year(year: i64) -> i64 {
    (year % 100 + 100) % 100
}

/// The week of the year, Sunday (`shift` 7) or Monday (`shift` 6) based.
fn week_of_year(t: &libc::tm, shift: i64) -> i64 {
    (i64::from(t.tm_yday) + 7 - (i64::from(t.tm_wday) + shift) % 7) / 7
}

/// The ISO 8601 week-based year and week, or None for a weekday or day of
/// the year the console refuses to convert.
fn iso_week(t: &libc::tm) -> Option<(i64, i64)> {
    let year_len = if is_leap_year_sum(i64::from(t.tm_year), 1900) { 366 } else { 365 };
    if t.tm_yday < 0 || i64::from(t.tm_yday) >= year_len || t.tm_wday < -6 {
        return None;
    }
    let mut base: i64 = 1900;
    let mut yday = i64::from(t.tm_yday);
    loop {
        let length = if is_leap_year_sum(i64::from(t.tm_year), base) { 366 } else { 365 };
        let bottom = (yday + 11 - i64::from(t.tm_wday)) % 7 - 3;
        let mut top = bottom - length % 7;
        if top < -3 {
            top += 7;
        }
        top += length;
        if yday >= top {
            return Some((i64::from(t.tm_year) + base + 1, 1));
        }
        if yday >= bottom {
            return Some((i64::from(t.tm_year) + base, 1 + (yday - bottom) / 7));
        }
        base -= 1;
        yday += if is_leap_year_sum(i64::from(t.tm_year), base) { 366 } else { 365 };
    }
}

/// `%z`: the guest's `tm_gmtoff`, one hour added east for a positive
/// `tm_isdst` (an unknown `-1` adds nothing) — measured on hardware; the
/// host zone is not consulted.
fn add_zone_offset(buffer: &mut FormatBuffer, t: &libc::tm) {
    let mut east = t.tm_gmtoff as i64 + i64::from(t.tm_isdst > 0) * 3600;
    let sign = if east < 0 { '-' } else { '+' };
    if east < 0 {
        east = -east;
    }
    buffer.add_text(&format!("{sign}{:02}{:02}", east / 3600, east / 60 % 60));
}

/// One conversion; whether the console knows it (an unknown one is copied
/// through as `%` and the character).
fn add_conversion(buffer: &mut FormatBuffer, conversion: u8, t: &libc::tm) -> bool {
    let year = i64::from(t.tm_year) + 1900;
    match conversion {
        b'a' => buffer.add_text(table_name(t.tm_wday, &STRFTIME_WEEKDAYS, &STRFTIME_WEEKDAYS)),
        b'A' => buffer.add_text(table_name(t.tm_wday, &STRFTIME_WEEKDAYS_FULL, &STRFTIME_WEEKDAYS)),
        b'b' | b'h' => buffer.add_text(table_name(t.tm_mon, &STRFTIME_MONTHS, &STRFTIME_MONTHS)),
        b'B' => buffer.add_text(table_name(t.tm_mon, &STRFTIME_MONTHS_FULL, &STRFTIME_MONTHS)),
        b'c' => format_time(buffer, b"%a %b %e %H:%M:%S %Y", t),
        b'C' => buffer.add_number(i64::from(t.tm_year) / 100 + 19, 100, 2, '0'),
        b'd' => buffer.add_number(i64::from(t.tm_mday), 100, 2, '0'),
        b'D' | b'x' => format_time(buffer, b"%m/%d/%y", t),
        b'e' => buffer.add_number(i64::from(t.tm_mday), 100, 2, ' '),
        b'F' => format_time(buffer, b"%Y-%m-%d", t),
        b'g' => match iso_week(t) {
            Some((y, _)) => buffer.add_number(two_digit_year(y), 100, 2, '0'),
            None => buffer.bad = true,
        },
        b'G' => match iso_week(t) {
            Some((y, _)) => buffer.add_number(y, 10000, 4, '0'),
            None => buffer.bad = true,
        },
        b'H' => buffer.add_number(i64::from(t.tm_hour), 100, 2, '0'),
        b'I' => buffer.add_number((i64::from(t.tm_hour) + 11) % 12 + 1, 100, 2, '0'),
        b'j' => buffer.add_number(i64::from(t.tm_yday) + 1, 1000, 3, '0'),
        b'm' => buffer.add_number(i64::from(t.tm_mon) + 1, 100, 2, '0'),
        b'M' => buffer.add_number(i64::from(t.tm_min), 100, 2, '0'),
        b'n' => buffer.add_text("\n"),
        b'p' => buffer.add_text(if t.tm_hour < 12 { "AM" } else { "PM" }),
        b'r' => format_time(buffer, b"%I:%M:%S %p", t),
        b'R' => format_time(buffer, b"%H:%M", t),
        b'S' => buffer.add_number(i64::from(t.tm_sec), 100, 2, '0'),
        b't' => buffer.add_text("\t"),
        b'T' | b'X' => format_time(buffer, b"%H:%M:%S", t),
        b'u' => buffer.add_number(i64::from(if t.tm_wday == 0 { 7 } else { t.tm_wday }), 10, 0, ' '),
        b'U' => buffer.add_number(week_of_year(t, 7), 100, 2, '0'),
        b'V' => match iso_week(t) {
            Some((_, week)) => buffer.add_number(week, 100, 2, '0'),
            None => buffer.bad = true,
        },
        b'w' => buffer.add_number(i64::from(t.tm_wday), 10, 0, ' '),
        b'W' => buffer.add_number(week_of_year(t, 6), 100, 2, '0'),
        b'y' => buffer.add_number(two_digit_year(year), 100, 2, '0'),
        b'Y' => buffer.add_number(year, 10000, 4, '0'),
        b'z' => add_zone_offset(buffer, t),
        b'Z' => buffer.add_piece(&host_zone_name(false)),
        b'%' => buffer.add_text("%"),
        _ => return false,
    }
    true
}

/// The format string itself: literal text, a conversion after `%` (an `E`
/// or `O` before it stepped over), an unknown one copied through. A format
/// ending after the modifier is an error on the console; it is reported
/// and the rest of the format dropped.
fn format_time(buffer: &mut FormatBuffer, format: &[u8], t: &libc::tm) {
    let mut i = 0;
    while i < format.len() && !buffer.full {
        let c = format[i];
        i += 1;
        if c != b'%' || i >= format.len() {
            buffer.add_piece(&[c]);
            continue;
        }
        let mut conversion = format[i];
        i += 1;
        if conversion == b'E' || conversion == b'O' {
            let Some(&next) = format.get(i) else {
                log::warn!("strftime: the format ends after an E or O modifier");
                buffer.bad = true;
                break;
            };
            conversion = next;
            i += 1;
        }
        if !add_conversion(buffer, conversion, t) {
            buffer.add_piece(&[b'%', conversion]);
        }
    }
}

/// strftime(str, count, format, const struct tm*)
fn cb_strftime(ctx: &mut CpuContext) -> HleResult {
    let (dst, count, fmt, tm) = (ctx.rdi(), ctx.rsi() as usize, ctx.rdx(), ctx.rcx());
    if dst == 0 || fmt == 0 || tm == 0 {
        return 0;
    }
    let Some(t) = tm_from_guest(ctx, tm) else { return 0 };
    let Some(format) = ctx.read_cstr_bytes(fmt, MAX_CSTR) else { return 0 };
    let count = count.min(MAX_CSTR);
    let mut out = vec![0u8; count];
    let (len, full, bad) = {
        let mut buffer = FormatBuffer { out: &mut out, len: 0, full: false, bad: false };
        format_time(&mut buffer, &format, &t);
        (buffer.len, buffer.full, buffer.bad)
    };
    // A piece that did not fit (or no room left for the terminator) fails.
    if bad || full || len == out.len() {
        return 0;
    }
    out[len] = 0;
    ctx.write_bytes(dst, &out[..len + 1]);
    len as HleResult
}

/// The `asctime` text of a broken-down time: "Www Mmm dd hh:mm:ss yyyy\n"
/// with the fields the console prints (an out-of-range weekday or month
/// reads the first name, the numbers wrap at their width).
fn asctime_text(t: &libc::tm) -> String {
    format!(
        "{} {}{:3} {:02}:{:02}:{:02} {:04}\n",
        table_name(t.tm_wday, &STRFTIME_WEEKDAYS, &STRFTIME_WEEKDAYS),
        table_name(t.tm_mon, &STRFTIME_MONTHS, &STRFTIME_MONTHS),
        clamped_field(i64::from(t.tm_mday), 100),
        clamped_field(i64::from(t.tm_hour), 100),
        clamped_field(i64::from(t.tm_min), 100),
        clamped_field(i64::from(t.tm_sec), 100),
        clamped_field(i64::from(t.tm_year) + 1900, 10000),
    )
}

/// What ctime returns for a time localtime cannot convert.
const CTIME_UNKNOWN: &str = "??? ??? ?? ??:??:?? ????\n";
const CTIME_BUFFER: u64 = 64;

thread_local! {
    /// Per-thread guest buffer ctime returns.
    static CTIME_RESULT: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// The ctime text of `*timer`: the local time as asctime formats it, or
/// None when there is no time or localtime cannot convert it.
fn ctime_text(ctx: &CpuContext, timer: u64) -> Option<String> {
    let t = (timer != 0).then(|| ctx.read_u64(timer)).flatten()? as libc::time_t;
    let mut host: libc::tm = unsafe { std::mem::zeroed() };
    // SAFETY: both pointers are valid for the call.
    if unsafe { libc::localtime_r(&t, &mut host) }.is_null() {
        return None;
    }
    Some(asctime_text(&host))
}

/// ctime(const time_t*): asctime(localtime(timer)) in a per-thread buffer;
/// an unconvertible time gives "??? ??? ?? ??:??:?? ????\n" and EINVAL.
fn cb_ctime(ctx: &mut CpuContext) -> HleResult {
    let text = ctime_text(ctx, ctx.rdi()).unwrap_or_else(|| {
        crate::kernel::set_errno(ctx, EINVAL);
        CTIME_UNKNOWN.to_string()
    });
    let mut out = CTIME_RESULT.with(|c| c.get());
    if out == 0 {
        let Some(p) = crate::pthread::alloc_guest_block(ctx, CTIME_BUFFER) else { return 0 };
        CTIME_RESULT.with(|c| c.set(p));
        out = p;
    }
    let mut bytes = text.into_bytes();
    bytes.truncate(CTIME_BUFFER as usize - 1);
    bytes.push(0);
    if ctx.write_bytes(out, &bytes) { out as HleResult } else { 0 }
}

// ---------------------------------------------------------------------------
// Per-thread destructors (__cxa_thread_atexit of libSceLibcInternal)
// ---------------------------------------------------------------------------
//
// The game's libc forwards `__cxa_thread_atexit(dtor, obj, dso)` to
// `_sceLibcInternalThreadAtexit`, `exit` to `_sceLibcInternalThreadDtors`
// and `__cxa_finalize(dso)` to `_sceLibcInternalForceTlsDestructor(module)`.
// Every thread keeps its own list, newest entry first:
// - registration fails with -1 when `dso` lies in no loaded module, and
//   records the module that holds `dso`;
// - an entry runs only while its `dso` is still inside a loaded module and
//   still holds its own address (a live `__dso_handle`); otherwise it is
//   dropped without a call;
// - destructors run newest first in passes: entries a destructor registers
//   go to the next pass, and after four passes whatever is left is dropped;
// - forcing a module runs, in the same passes, only the calling thread's
//   entries of that module, and stops after a pass that found none.

/// Passes over a thread's destructor list before the rest is dropped.
const THREAD_DTOR_PASSES: u32 = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ThreadAtexitEntry {
    /// Registration order; entries registered during a pass are newer than
    /// the pass's boundary and wait for the next one.
    seq: u64,
    dtor: u64,
    obj: u64,
    dso: u64,
    /// Handle of the module holding `dso` when the entry was registered.
    module: i32,
}

#[derive(Default)]
struct ThreadAtexitList {
    /// Oldest first.
    entries: Vec<ThreadAtexitEntry>,
    next_seq: u64,
    /// Pass number and boundary of the thread-exit run in progress.
    exit_pass: Option<(u32, u64)>,
}

impl ThreadAtexitList {
    fn push(&mut self, dtor: u64, obj: u64, dso: u64, module: i32) {
        let seq = self.next_seq;
        self.next_seq += 1;
        self.entries.push(ThreadAtexitEntry { seq, dtor, obj, dso, module });
    }

    /// Remove and return the newest entry registered before `boundary`,
    /// limited to one module when `module` is given.
    fn take_before(&mut self, boundary: u64, module: Option<i32>) -> Option<ThreadAtexitEntry> {
        let i = self.entries.iter().rposition(|e| e.seq < boundary && module.is_none_or(|m| e.module == m))?;
        Some(self.entries.remove(i))
    }

    /// The next entry of the thread-exit run (every module), or None when
    /// the run is over; the run then forgets whatever is left.
    fn next_exit_entry(&mut self) -> Option<ThreadAtexitEntry> {
        loop {
            let (pass, boundary) = match self.exit_pass {
                Some(p) => p,
                None if self.entries.is_empty() => return None,
                None => *self.exit_pass.insert((1, self.next_seq)),
            };
            if let Some(e) = self.take_before(boundary, None) {
                return Some(e);
            }
            if self.entries.is_empty() || pass >= THREAD_DTOR_PASSES {
                self.entries.clear();
                self.exit_pass = None;
                return None;
            }
            self.exit_pass = Some((pass + 1, self.next_seq));
        }
    }
}

thread_local! {
    static THREAD_ATEXIT: std::cell::RefCell<ThreadAtexitList> = std::cell::RefCell::new(ThreadAtexitList::default());
}

fn with_thread_atexit<R>(f: impl FnOnce(&mut ThreadAtexitList) -> R) -> R {
    THREAD_ATEXIT.with(|l| f(&mut l.borrow_mut()))
}

/// Whether `dso` is still a live `__dso_handle`: inside a loaded module and
/// holding its own address.
fn dso_is_live(dso: u64) -> bool {
    if crate::modules::module_by_address(dso).is_none() {
        return false;
    }
    let mut word = [0u8; 8];
    crate::pthread::guest_vm().is_some_and(|vm| vm.read(dso, &mut word).is_ok()) && u64::from_le_bytes(word) == dso
}

/// _sceLibcInternalThreadAtexit(dtor, obj, dso) -> 0, or -1 when `dso` is
/// not inside a loaded module.
fn cb_cxa_thread_atexit(ctx: &mut CpuContext) -> HleResult {
    let (dtor, obj, dso) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    let Some(module) = crate::modules::module_by_address(dso) else { return -1 };
    with_thread_atexit(|l| l.push(dtor, obj, dso, module.handle));
    OK
}

/// _sceLibcInternalThreadDtors(): run the calling thread's destructors.
fn cb_thread_dtors(_ctx: &mut CpuContext) -> HleResult {
    run_thread_atexit_destructors();
    OK
}

/// _sceLibcInternalForceTlsDestructor(module): run the calling thread's
/// destructors registered for `module` (its `__cxa_finalize`), newest first,
/// in up to four passes; the other modules' entries stay.
fn cb_force_tls_destructor(ctx: &mut CpuContext) -> HleResult {
    let module = ctx.rdi() as i32;
    for _ in 0..THREAD_DTOR_PASSES {
        let boundary = with_thread_atexit(|l| l.next_seq);
        let mut found = false;
        while let Some(e) = with_thread_atexit(|l| l.take_before(boundary, Some(module))) {
            found = true;
            if e.dtor != 0 && dso_is_live(e.dso) {
                ps5cpu::exec::call_guest_nested(e.dtor, [e.obj, 0, 0, 0, 0, 0]);
            }
        }
        if !found {
            break;
        }
    }
    OK
}

/// Run the calling thread's registered destructors, last registered first.
/// Must be called on the guest thread, from an HLE dispatch.
pub fn run_thread_atexit_destructors() {
    while let Some((dtor, obj)) = pop_thread_atexit_destructor() {
        ps5cpu::exec::call_guest_nested(dtor, [obj, 0, 0, 0, 0, 0]);
    }
}

/// Take the calling thread's next destructor `(dtor, obj)` to run at
/// thread exit, skipping entries whose module is gone; None ends the run
/// (call until None). For exit paths that run outside a dispatch.
pub fn pop_thread_atexit_destructor() -> Option<(u64, u64)> {
    loop {
        let e = with_thread_atexit(|l| l.next_exit_entry())?;
        if e.dtor != 0 && dso_is_live(e.dso) {
            return Some((e.dtor, e.obj));
        }
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Aligned allocations from the HLE heap (bound like malloc and free).
const ALIGNED_ALLOCATOR_EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("Ujf3KzMvRmI", "memalign", cb_memalign),
    ("2Btkg8k24Zg", "aligned_alloc", cb_aligned_alloc),
    ("cVSk9y8URbc", "posix_memalign", cb_posix_memalign),
    ("NDcSfcYZRC8", "malloc_usable_size", cb_malloc_usable_size),
];

/// Served only when no loaded libc module exports them.
const FALLBACK_EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("5Xa2ACNECdo", "strcpy_s", cb_strcpy_s),
    ("6sJWiWSRuqk", "strncpy", cb_strncpy),
    ("Ls4tzzhimqQ", "strcat", cb_strcat),
    ("ob5xAW4ln-0", "strchr", cb_strchr),
    ("9yDWMxEFdJU", "strrchr", cb_strrchr),
    ("viiwFMaNamA", "strstr", cb_strstr),
    ("mXlxhmLNMPg", "strtol", cb_strtol),
    ("QxmSHBCuKTk", "strtoul", cb_strtoul),
    ("zlfEH8FmyUA", "_Stoul", cb_strtoul),
    ("fPxypibz2MY", "atoi", cb_atoi),
    ("+my9jdHCMIQ", "atol", cb_atol),
    ("fLcU5G6Qrjs", "atoll", cb_atol),
    ("gYFKAMoNEfo", "tzset", cb_tzset),
    ("-nvxBWa0iDs", "gethostname", cb_gethostname),
    ("wLlFkwG9UcQ", "time", cb_time),
    ("-VVn74ZyhEs", "difftime", cb_difftime),
    ("1mecP7RgI2A", "gmtime", cb_gmtime),
    ("efhK-YSUYYQ", "localtime", cb_localtime),
    ("5bBacGLyLOs", "gmtime_s", cb_gmtime_s),
    ("fiiNDnNBKVY", "localtime_s", cb_localtime_s),
    ("n7AepwR0s34", "mktime", cb_mktime),
    ("Av3zjWi64Kw", "strftime", cb_strftime),
    ("0uAUs3hYuG4", "ctime", cb_ctime),
    ("jbz9I9vkqkk", "vsprintf", cb_vsprintf),
    // vsnprintf_s formats as vsnprintf.
    ("rWSuTWY2JN0", "vsnprintf_s", cb_vsnprintf),
    ("tfNbpqL3D0M", "vsscanf_s", cb_vsscanf_s),
    ("tQNolUV1q5A", "swscanf_s", cb_swscanf_s),
    ("nJz16JE1txM", "swprintf", cb_swprintf),
    ("w1NxRBQqfmQ", "printf_s", cb_printf_s),
    ("YfJUGNPkbK4", "vprintf_s", cb_vprintf_s),
];

/// Register all libc HLE exports into the registry.
pub fn register(reg: &mut HleRegistry) {
    reg.register_fn("gQX+4GDQjpM", "malloc", "libc", cb_malloc);
    reg.register_fn("2X5agFjKxMc", "calloc", "libc", cb_calloc);
    reg.register_fn("Y7aJ1uydPMo", "realloc", "libc", cb_realloc);
    reg.register_fn("tIhsqj0qsFE", "free", "libc", cb_free);
    // The rest of the allocator family, so a game without its own libc
    // never mixes these with a native allocator.
    for &(nid, name, cb) in ALIGNED_ALLOCATOR_EXPORTS {
        reg.register_fn(nid, name, "libc", cb);
    }
    reg.register_fn("Q3VBxCXhUHs", "memcpy", "libc", cb_memcpy);
    reg.register_fn("+P6FRGH4LfA", "memmove", "libc", cb_memmove);
    reg.register_fn("8zTFvBIAIN8", "memset", "libc", cb_memset);
    reg.register_fn("DfivPArhucg", "memcmp", "libc", cb_memcmp);
    reg.register_fn("j4ViWNHEgww", "strlen", "libc", cb_strlen);
    reg.register_fn("5jNubw4vlAA", "strnlen", "libc", cb_strnlen);
    reg.register_fn("Ovb2dSJOAuE", "strcmp", "libc", cb_strcmp);
    reg.register_fn("aesyjrHVWy4", "strncmp", "libc", cb_strncmp);
    reg.register_fn("kiZSXIWd9vg", "strcpy", "libc", cb_strcpy);
    reg.register_fn("Ou3iL1abvng", "__stack_chk_fail", "libc", cb_stack_chk_fail);
    reg.register_fn("Q2V+iqvjgC0", "vsnprintf", "libc", cb_vsnprintf);
    reg.register_fallback_fn("eLdDw6l0-bU", "snprintf", "libc", cb_snprintf);
    reg.register_fallback_fn("GMpvxPFW924", "vprintf", "libc", cb_vprintf);
    reg.register_fn("hcuQgD53UxM", "printf", "libc", cb_printf);
    reg.register_fallback_fn("YQ0navp+YIc", "puts", "libc", cb_puts);
    reg.register_fn("L1SBTkC+Cvw", "abort", "libc", cb_abort);
    reg.register_fallback_fn("uMei1W9uyNo", "exit", "libc", cb_exit);
    // Wide-string / wide-printf family used by game logging.
    reg.register_fn("WkkeywLJcgU", "wcslen", "libc", cb_wcslen);
    reg.register_fn("u0XOsuOmOzc", "vswprintf", "libc", cb_vswprintf);
    reg.register_fn("oDoV9tyHTbA", "vswprintf_s", "libc", cb_vswprintf_s);
    reg.register_fn("Im55VJ-Bekc", "swprintf_s", "libc", cb_swprintf_s);

    // Heap trace info — called by libc.prx DT_INIT.
    // Fills an Info struct with pointers to mstate tracking arrays.
    for &(nid, name, cb) in FALLBACK_EXPORTS {
        reg.register_fallback_fn(nid, name, "libc", cb);
    }
    reg.register_fn("qBS714-Jr3g", "_sceLibcInternalThreadAtexit", "libSceLibcInternal", cb_cxa_thread_atexit);
    reg.register_fn("OQ-dzhlnM28", "_sceLibcInternalThreadDtors", "libSceLibcInternal", cb_thread_dtors);
    reg.register_fn("tB59hFLH3SA", "_sceLibcInternalForceTlsDestructor", "libSceLibcInternal", cb_force_tls_destructor);

    reg.register_fn("NWtTN10cJzE", "sceLibcHeapGetTraceInfo", "libc", cb_heap_get_trace_info);
}

/// Static mstate tracking arrays for heap trace info.
/// These live at fixed guest addresses so libc.prx can reference them.
/// Heap trace storage: a u64 mspace id mask followed by a 64-entry
/// mstate table, zero-initialised, shared by every caller.
static HEAP_TRACE_STORAGE: Mutex<u64> = Mutex::new(0);
const HEAP_TRACE_INFO_SIZE: u64 = 32;
/// The native libc indexes mstate_table[0..=64]: 65 slots.
const HEAP_TRACE_TABLE_ENTRIES: u64 = 65;

/// sceLibcHeapGetTraceInfo(info*): info = { u64 size (32); u32; u32 flags;
/// u64* mspace_id_mask @0x10; u64* mstate_table @0x18 }.
fn cb_heap_get_trace_info(ctx: &mut CpuContext) -> HleResult {
    const EINVAL: HleResult = 0x8002_0016u32 as i32 as i64;
    const EFAULT: HleResult = 0x8002_000Eu32 as i32 as i64;
    let info = ctx.rdi();
    if info == 0 || ctx.read_u64(info) != Some(HEAP_TRACE_INFO_SIZE) {
        return EINVAL;
    }
    let storage = {
        let mut s = HEAP_TRACE_STORAGE.lock().unwrap();
        if *s == 0 {
            match crate::pthread::alloc_guest_block(ctx, 8 + HEAP_TRACE_TABLE_ENTRIES * 8) {
                Some(addr) => {
                    ctx.write_bytes(addr, &vec![0u8; (8 + HEAP_TRACE_TABLE_ENTRIES * 8) as usize]);
                    *s = addr;
                }
                None => return EFAULT,
            }
        }
        *s
    };
    if !ctx.write_u32(info + 0x0C, 0) || !ctx.write_u64(info + 0x10, storage) || !ctx.write_u64(info + 0x18, storage + 8) {
        return EFAULT;
    }
    OK
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CpuRegister::{Rdi, Rdx, Rsi};

    fn exit_run(l: &mut ThreadAtexitList, mut on_call: impl FnMut(&mut ThreadAtexitList, u64)) -> Vec<u64> {
        let mut order = Vec::new();
        while let Some(e) = l.next_exit_entry() {
            order.push(e.obj);
            on_call(l, e.obj);
        }
        order
    }

    /// Thread exit: newest first; a destructor's own registrations run in
    /// the next pass; after four passes the rest is dropped.
    #[test]
    fn thread_atexit_exit_passes() {
        let mut l = ThreadAtexitList::default();
        for obj in 1..=3 {
            l.push(0x10, obj, 0x20, 1);
        }
        assert_eq!(exit_run(&mut l, |_, _| {}), vec![3, 2, 1]);
        assert!(l.entries.is_empty() && l.exit_pass.is_none());

        // Every destructor registers one more: passes 1..4 run, then stop.
        l.push(0x10, 100, 0x20, 1);
        let order = exit_run(&mut l, |l, obj| l.push(0x10, obj + 1, 0x20, 1));
        assert_eq!(order, vec![100, 101, 102, 103]);
        assert!(l.entries.is_empty(), "the fifth registration is dropped");

        // Two registered in one pass run newest first in the next.
        l.push(0x10, 1, 0x20, 1);
        let order = exit_run(&mut l, |l, obj| {
            if obj == 1 {
                l.push(0x10, 10, 0x20, 1);
                l.push(0x10, 11, 0x20, 1);
            }
        });
        assert_eq!(order, vec![1, 11, 10]);
        // A new list starts after a finished run.
        l.push(0x10, 7, 0x20, 1);
        assert_eq!(exit_run(&mut l, |_, _| {}), vec![7]);
    }

    /// Forcing a module takes only that module's entries, newest first,
    /// and leaves the others in order.
    #[test]
    fn thread_atexit_force_one_module() {
        let mut l = ThreadAtexitList::default();
        l.push(0x10, 1, 0x20, 5);
        l.push(0x10, 2, 0x30, 6);
        l.push(0x10, 3, 0x20, 5);
        let boundary = l.next_seq;
        l.push(0x10, 4, 0x20, 5);
        assert_eq!(l.take_before(boundary, Some(5)).map(|e| e.obj), Some(3));
        assert_eq!(l.take_before(boundary, Some(5)).map(|e| e.obj), Some(1));
        assert_eq!(l.take_before(boundary, Some(5)), None);
        assert_eq!(l.take_before(boundary, Some(7)), None);
        assert_eq!(exit_run(&mut l, |_, _| {}), vec![4, 2]);
    }

    /// Registration needs a `dso` inside a loaded module; none is loaded
    /// here, so it fails with -1 and records nothing; with no entries both
    /// runs do nothing.
    #[test]
    fn thread_atexit_rejects_dso_outside_modules() {
        let vm = ps5vm::VirtualMemory::new();
        let mut ctx = CpuContext::new(&vm);
        ctx.set(Rdi, 0x1000);
        ctx.set(Rsi, 0x2000);
        ctx.set(Rdx, 0);
        assert_eq!(cb_cxa_thread_atexit(&mut ctx), -1);
        assert_eq!(with_thread_atexit(|l| l.entries.len()), 0);
        assert_eq!(cb_thread_dtors(&mut ctx), OK);
        ctx.set(Rdi, 3);
        assert_eq!(cb_force_tls_destructor(&mut ctx), OK);
        assert_eq!(pop_thread_atexit_destructor(), None);
    }

    #[test]
    fn libc_internal_nids() {
        for (nid, name) in [
            ("qBS714-Jr3g", "_sceLibcInternalThreadAtexit"),
            ("OQ-dzhlnM28", "_sceLibcInternalThreadDtors"),
            ("tB59hFLH3SA", "_sceLibcInternalForceTlsDestructor"),
        ] {
            assert_eq!(ps5nid::compute_nid(name), nid, "{name}");
        }
    }

    /// sceLibcHeapGetTraceInfo: a mask word and a zeroed 65-slot table
    /// (the native libc writes slot 64 too); the second u32 is cleared.
    #[test]
    fn heap_trace_info_table() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5C27_4000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.write_bytes(BASE, &[0xFF; 32]);
        ctx.write_u64(BASE, 32);
        ctx.set(Rdi, BASE);
        assert_eq!(cb_heap_get_trace_info(&mut ctx), OK);
        assert_eq!(ctx.read_u32(BASE + 0x0C), Some(0));
        let (mask, table) = (ctx.read_u64(BASE + 0x10).unwrap(), ctx.read_u64(BASE + 0x18).unwrap());
        assert_eq!(table, mask + 8);
        assert_eq!(ctx.read_bytes(table, 65 * 8).unwrap(), vec![0; 65 * 8]);
        assert!(ctx.write_u64(table + 64 * 8, 0) && ctx.read_u64(table + 64 * 8) == Some(0));
        ctx.write_u64(BASE, 24);
        assert_eq!(cb_heap_get_trace_info(&mut ctx), 0x8002_0016u32 as i32 as i64);
    }

    #[test]
    fn string_functions_at_end_of_mapping() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5C27_0000_0000;
        vm.map(BASE, 0x2000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        // A string that ends a few bytes before the unmapped page that
        // follows the mapping, like a path built at the top of a stack.
        let src = BASE + 0x2000 - 0x10;
        ctx.write_bytes(src, b"/app0/\xe9data\0");
        ctx.set(Rdi, src);
        assert_eq!(cb_strlen(&mut ctx), 11);
        ctx.set(Rsi, 4);
        assert_eq!(cb_strnlen(&mut ctx), 4);

        let dst = BASE + 0x100;
        ctx.write_bytes(dst, &[0xAA; 16]);
        ctx.set(Rdi, dst);
        ctx.set(Rsi, src);
        assert_eq!(cb_strcpy(&mut ctx), dst as i64);
        assert_eq!(ctx.read_bytes(dst, 12).unwrap(), b"/app0/\xe9data\0");
        assert_eq!(cb_strcmp(&mut ctx), 0);

        ctx.write_bytes(dst, b"/app0/\0");
        assert!(cb_strcmp(&mut ctx) < 0);
        ctx.set(Rdx, 6);
        assert_eq!(cb_strncmp(&mut ctx), 0);
        ctx.set(Rdx, u64::MAX);
        assert!(cb_strncmp(&mut ctx) < 0);
        // Bytes compare unsigned.
        ctx.write_bytes(dst, b"/app0/a\0");
        assert!(cb_strcmp(&mut ctx) < 0);
    }

    #[test]
    fn fallback_nids_match_names() {
        for &(nid, name, _) in FALLBACK_EXPORTS.iter().chain(ALIGNED_ALLOCATOR_EXPORTS) {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    /// Every block is 32-byte aligned (small sizes, large ones, a grown
    /// realloc) and holds its bytes; posix_memalign / memalign honour larger
    /// alignments and refuse what the heap cannot hold.
    #[test]
    fn heap_blocks_are_32_byte_aligned_and_aligned_requests_fail_cleanly() {
        let vm = ps5vm::VirtualMemory::new();
        let mut ctx = CpuContext::new(&vm);
        let call = |ctx: &mut CpuContext, cb: HleCallback, args: &[u64]| {
            for (r, v) in [Rdi, Rsi, Rdx].iter().zip(args) {
                ctx.set(*r, *v);
            }
            cb(ctx) as u64
        };
        let mut blocks: Vec<(u64, u64)> = (1..=600u64).map(|n| (call(&mut ctx, cb_malloc, &[n]), n)).collect();
        for n in [4000u64, 70000, 0x30000] {
            blocks.push((call(&mut ctx, cb_malloc, &[n]), n));
        }
        let small = call(&mut ctx, cb_malloc, &[8]);
        ctx.write_bytes(small, b"8 bytes!");
        let grown = call(&mut ctx, cb_realloc, &[small, 333]);
        assert_eq!(ctx.read_bytes(grown, 8).unwrap(), b"8 bytes!");
        blocks.push((grown, 333));
        for (i, &(p, n)) in blocks.iter().enumerate() {
            assert_ne!(p, 0);
            assert_eq!(p & 31, 0, "block of {n} bytes at {p:#x}");
            ctx.write_bytes(p, &vec![i as u8; n as usize]);
        }
        for (i, &(p, n)) in blocks.iter().enumerate() {
            assert!(ctx.read_bytes(p, n as usize).unwrap().iter().all(|&b| b == i as u8));
            call(&mut ctx, cb_free, &[p]);
        }
        // malloc(0) and calloc(0, n) still return unique blocks.
        let (z1, z2) = (call(&mut ctx, cb_malloc, &[0]), call(&mut ctx, cb_calloc, &[0, 4]));
        assert!(z1 != 0 && z2 != 0 && z1 != z2);
        assert_eq!(call(&mut ctx, cb_calloc, &[u64::MAX, 2]), 0);
        assert_eq!(call(&mut ctx, cb_malloc, &[u64::MAX]), 0);

        let out = call(&mut ctx, cb_malloc, &[16]);
        ctx.write_u64(out, 0);
        assert_eq!(call(&mut ctx, cb_posix_memalign, &[out, 256, 99]), 0);
        let aligned = ctx.read_u64(out).unwrap();
        assert!(aligned != 0 && aligned % 256 == 0);
        call(&mut ctx, cb_free, &[aligned]);
        ctx.write_u64(out, 0);
        assert_eq!(call(&mut ctx, cb_posix_memalign, &[out, 256, u64::MAX]), ENOMEM as u64);
        assert_eq!(call(&mut ctx, cb_posix_memalign, &[out, 4096, u64::MAX - 4096]), ENOMEM as u64);
        assert_eq!(ctx.read_u64(out), Some(0));
        assert_eq!(call(&mut ctx, cb_posix_memalign, &[out, 3, 16]), EINVAL as u64);
        assert_eq!(call(&mut ctx, cb_posix_memalign, &[out, 4, 16]), EINVAL as u64);
        assert_eq!(call(&mut ctx, cb_posix_memalign, &[0, 64, 16]), EINVAL as u64);
        assert_eq!(ctx.read_u64(out), Some(0));
        let page = call(&mut ctx, cb_memalign, &[4096, 10]);
        assert!(page != 0 && page % 4096 == 0);
        let wide = call(&mut ctx, cb_aligned_alloc, &[64, 64]);
        assert!(wide != 0 && wide % 64 == 0);
        assert_eq!(call(&mut ctx, cb_memalign, &[3, 16]), 0);
        for p in [page, wide, z1, z2, out] {
            call(&mut ctx, cb_free, &[p]);
        }
    }

    /// malloc_usable_size: at least the requested size for small and large
    /// blocks, 0 for NULL, a freed block or an address inside a block.
    #[test]
    fn malloc_usable_size_covers_the_request() {
        let vm = ps5vm::VirtualMemory::new();
        let mut ctx = CpuContext::new(&vm);
        let call = |ctx: &mut CpuContext, cb: HleCallback, arg: u64| {
            ctx.set(Rdi, arg);
            cb(ctx) as u64
        };
        let small = call(&mut ctx, cb_malloc, 16);
        let large = call(&mut ctx, cb_malloc, 100_000);
        let odd = call(&mut ctx, cb_malloc, 33);
        assert_eq!(call(&mut ctx, cb_malloc_usable_size, small), 32, "rounded to the heap alignment");
        assert_eq!(call(&mut ctx, cb_malloc_usable_size, large), 100_000);
        let u = call(&mut ctx, cb_malloc_usable_size, odd);
        assert!((33..33 + DEFAULT_ALIGN).contains(&u), "usable {u}");
        assert_eq!(call(&mut ctx, cb_malloc_usable_size, 0), 0);
        assert_eq!(call(&mut ctx, cb_malloc_usable_size, large + 8), 0);
        call(&mut ctx, cb_free, large);
        assert_eq!(call(&mut ctx, cb_malloc_usable_size, large), 0, "freed block");
        call(&mut ctx, cb_free, small);
        call(&mut ctx, cb_free, odd);
    }

    /// FreeBSD's strtol family knows no binary literals: with base 0 or 2,
    /// "0b101" reads "0" and stops at the 'b'.
    #[test]
    fn strtol_family_stops_at_a_binary_prefix() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5D57_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let (text, end) = (BASE, BASE + 0x100);
        let parse = |ctx: &mut CpuContext, s: &[u8], base: u64, cb: HleCallback| {
            ctx.write_bytes(text, s);
            ctx.write_bytes(text + s.len() as u64, &[0]);
            ctx.set(Rdi, text);
            ctx.set(Rsi, end);
            ctx.set(Rdx, base);
            let v = cb(ctx);
            (v, ctx.read_u64(end).unwrap() - text)
        };
        for cb in [cb_strtol as HleCallback, cb_strtoul] {
            assert_eq!(parse(&mut ctx, b"0b101", 0, cb), (0, 1));
            assert_eq!(parse(&mut ctx, b"0b101", 2, cb), (0, 1));
            assert_eq!(parse(&mut ctx, b" -0B11", 0, cb), (0, 3));
            assert_eq!(parse(&mut ctx, b" +0b1!", 2, cb), (0, 3));
            assert_eq!(parse(&mut ctx, b"0b2", 0, cb), (0, 1));
            assert_eq!(parse(&mut ctx, b"0b101", 16, cb), (0xb101, 5));
            // A 0x prefix without a hex digit reads "0" and stops at the x.
            assert_eq!(parse(&mut ctx, b" -0xz!", 0, cb), (0, 3));
            assert_eq!(parse(&mut ctx, b" +0Xg!", 0, cb), (0, 3));
            assert_eq!(parse(&mut ctx, b"0xg", 16, cb), (0, 1));
            assert_eq!(parse(&mut ctx, b"0x", 16, cb), (0, 1));
            assert_eq!(parse(&mut ctx, b"0x!", 16, cb), (0, 1));
        }
        // atol / atoll: base 10 on 64 bits.
        let atol = |ctx: &mut CpuContext, s: &[u8]| {
            ctx.write_bytes(text, s);
            ctx.write_bytes(text + s.len() as u64, &[0]);
            ctx.set(Rdi, text);
            cb_atol(ctx)
        };
        assert_eq!(atol(&mut ctx, b" \t-4294967296tail"), -4294967296);
        assert_eq!(atol(&mut ctx, b"9223372036854775807"), i64::MAX);
        assert_eq!(atol(&mut ctx, b"+12"), 12);
        assert_eq!(atol(&mut ctx, b"0x10"), 0);
        assert_eq!(atol(&mut ctx, b"010"), 10);
        assert_eq!(atol(&mut ctx, b""), 0);
        assert_eq!(atol(&mut ctx, b"-9223372036854775808"), i64::MIN);
        assert_eq!(atol(&mut ctx, b"4294967297x"), 4294967297);
        assert_eq!(atol(&mut ctx, b"  -0012"), -12);
        assert_eq!(atol(&mut ctx, b"z1"), 0);
        assert_eq!(cb_tzset(&mut ctx), 0);
        // The guest's unsigned long is 64 bits (_Stoul shares strtoul).
        assert_eq!(parse(&mut ctx, b"4294967296", 10, cb_strtoul), (4294967296, 10));
        assert_eq!(parse(&mut ctx, b"18446744073709551615", 10, cb_strtoul), (-1, 20));
        assert_eq!(parse(&mut ctx, b"-1", 10, cb_strtoul), (-1, 2));
        vm.unmap(BASE, 0x1000).unwrap();
    }

    /// swprintf formats guest wide strings as 16-bit units: widths and
    /// precisions count units, %s converts UTF-8, truncation and rejected
    /// formats return -1.
    #[test]
    fn swprintf_formats_16_bit_units() {
        use crate::context::CpuRegister::{Rcx, Rsp, R8, R9};
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5E62_0000_0000;
        vm.map(BASE, 0x4000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.set(Rsp, BASE + 0x3000);
        let mut next = BASE + 0x1000;
        let mut put = |ctx: &CpuContext, bytes: &[u8]| {
            let at = next;
            ctx.write_bytes(at, bytes);
            next += (bytes.len() as u64 + 15) & !15;
            at
        };
        let wide = |s: &str| -> Vec<u8> { s.encode_utf16().chain([0]).flat_map(u16::to_le_bytes).collect() };
        let buf = BASE;
        let fill = |ctx: &CpuContext| ctx.write_bytes(buf, &[b'x', 0].repeat(16));
        let units = |ctx: &CpuContext, n: usize| -> Vec<u16> {
            ctx.read_bytes(buf, n * 2).unwrap().chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
        };
        let text = |ctx: &CpuContext| String::from_utf16_lossy(&ctx.read_wide_string(buf, 64).unwrap());
        let call = |ctx: &mut CpuContext, cb: HleCallback, dst: u64, size: u64, fmt: u64, args: &[u64]| {
            for (r, v) in [Rdi, Rsi, Rdx, Rcx, R8, R9].iter().zip([dst, size, fmt].iter().chain(args)) {
                ctx.set(*r, *v);
            }
            cb(ctx)
        };
        let f_d_ls = put(&ctx, &wide("%d-%ls"));
        let ab = put(&ctx, &wide("ab"));
        fill(&ctx);
        assert_eq!(call(&mut ctx, cb_swprintf, buf, 8, f_d_ls, &[42, ab]), 5);
        assert_eq!(text(&ctx), "42-ab");
        assert_eq!(units(&ctx, 7)[6], b'x' as u16, "nothing past the terminator");
        let f_cjk = put(&ctx, &wide("\u{4e2d}%ls%s"));
        let wen = put(&ctx, &wide("\u{6587}"));
        let e_acute = put(&ctx, b"\xc3\xa9\0");
        assert_eq!(call(&mut ctx, cb_swprintf, buf, 8, f_cjk, &[wen, e_acute]), 3);
        assert_eq!(text(&ctx), "\u{4e2d}\u{6587}\u{e9}");
        let f_s = put(&ctx, &wide("%s"));
        let emoji = put(&ctx, b"\xf0\x9f\x98\x80\0");
        assert_eq!(call(&mut ctx, cb_swprintf, buf, 8, f_s, &[emoji]), 2);
        assert_eq!(text(&ctx), "\u{1F600}");
        let empty = put(&ctx, &wide(""));
        assert_eq!(call(&mut ctx, cb_swprintf, buf, 8, empty, &[]), 0);
        assert_eq!(units(&ctx, 1)[0], 0);
        let f_ls = put(&ctx, &wide("%ls"));
        let abcd = put(&ctx, &wide("abcd"));
        fill(&ctx);
        assert_eq!(call(&mut ctx, cb_swprintf, buf, 3, f_ls, &[abcd]), -1);
        assert_eq!(text(&ctx), "ab");
        assert_eq!(units(&ctx, 4)[3], b'x' as u16);
        let abc = put(&ctx, &wide("abc"));
        assert_eq!(call(&mut ctx, cb_swprintf, buf, 3, abc, &[]), -1);
        fill(&ctx);
        assert_eq!(call(&mut ctx, cb_swprintf, buf, 0, abc, &[]), -1);
        assert_eq!(units(&ctx, 1)[0], b'x' as u16, "a zero size writes nothing");
        assert_eq!(call(&mut ctx, cb_swprintf, 0, 8, abc, &[]), -1);
        assert_eq!(call(&mut ctx, cb_swprintf, buf, 5, f_ls, &[abcd]), 4);
        assert_eq!(text(&ctx), "abcd");
        let f_pad = put(&ctx, &wide("%5ls|%-3c|%lc"));
        let wide_word = put(&ctx, &wide("wide"));
        assert_eq!(call(&mut ctx, cb_swprintf, buf, 16, f_pad, &[wide_word, b'z' as u64, 0x20ac]), 11);
        assert_eq!(text(&ctx), " wide|z  |\u{20ac}");
        let f_q = put(&ctx, &wide("%q"));
        fill(&ctx);
        assert_eq!(call(&mut ctx, cb_swprintf, buf, 16, f_q, &[]), -1);
        assert_eq!(units(&ctx, 1)[0], 0, "a rejected format leaves an empty string");

        // Precisions of %s count wide units, never splitting a pair.
        let e_grave = put(&ctx, b"\xc3\xa9\xc3\xa8\0");
        let emoji_x = put(&ctx, b"\xf0\x9f\x98\x80x\0");
        let abcd_n = put(&ctx, b"abcd\0");
        for (fmt, arg, expected) in [
            ("%.2s", e_grave, "\u{e9}\u{e8}"),
            ("%.1s", e_grave, "\u{e9}"),
            ("%.0s", e_acute, ""),
            ("%.3s", e_acute, "\u{e9}"),
            ("%s", e_grave, "\u{e9}\u{e8}"),
            ("%.2s", abcd_n, "ab"),
            ("%4.2s", e_grave, "  \u{e9}\u{e8}"),
            ("%-4.2s", e_grave, "\u{e9}\u{e8}  "),
            ("%.2s", emoji_x, "\u{1F600}"),
            ("%.1s", emoji_x, ""),
            ("%.3s", emoji_x, "\u{1F600}x"),
            ("%s", 0, "(null)"),
        ] {
            let f = put(&ctx, &wide(fmt));
            let n = expected.encode_utf16().count() as HleResult;
            assert_eq!(call(&mut ctx, cb_swprintf, buf, 32, f, &[arg]), n, "{fmt}");
            assert_eq!(text(&ctx), expected, "{fmt}");
        }

        // %n counts units; a width, an L length or a null pointer is rejected.
        let count = put(&ctx, &[0xAA; 8]);
        let f_n = put(&ctx, &wide("ab%ncd"));
        assert_eq!(call(&mut ctx, cb_swprintf, buf, 16, f_n, &[count]), 4);
        assert_eq!(ctx.read_u32(count), Some(2));
        let f_sn = put(&ctx, &wide("%s%n!"));
        assert_eq!(call(&mut ctx, cb_swprintf, buf, 16, f_sn, &[emoji, count]), 3);
        assert_eq!(ctx.read_u32(count), Some(2));
        for fmt in ["a%5n", "a%Ln", "a%n"] {
            let f = put(&ctx, &wide(fmt));
            let arg = if fmt == "a%n" { 0 } else { count };
            assert_eq!(call(&mut ctx, cb_swprintf, buf, 16, f, &[arg]), -1, "{fmt}");
        }
        // The _s variants reject %n and null strings.
        assert_eq!(call(&mut ctx, cb_swprintf_s, buf, 16, f_n, &[count]), -1);
        assert_eq!(units(&ctx, 1)[0], 0);
        let f_a_ls = put(&ctx, &wide("a%ls"));
        assert_eq!(call(&mut ctx, cb_swprintf_s, buf, 16, f_a_ls, &[0]), -1);
        assert_eq!(call(&mut ctx, cb_swprintf, buf, 16, f_a_ls, &[0]), 7);
        assert_eq!(text(&ctx), "a(null)");
        vm.unmap(BASE, 0x4000).unwrap();
    }

    /// The guest tm is FreeBSD's 56-byte layout: gmtime fills tm_gmtoff 0
    /// and tm_zone "UTC" from the day count for any year that fits an int.
    #[test]
    fn gmtime_writes_the_56_byte_guest_tm() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5D58_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let (timer, out) = (BASE, BASE + 0x100);
        // (time, year, mon, mday, hour, min, sec, wday, yday)
        let cases: [(i64, [i32; 8]); 7] = [
            (0, [70, 0, 1, 0, 0, 0, 4, 0]),
            (-1, [69, 11, 31, 23, 59, 59, 3, 364]),
            (-31536000, [69, 0, 1, 0, 0, 0, 3, 0]),
            (951782400, [100, 1, 29, 0, 0, 0, 2, 59]),
            (32535216000, [1101, 0, 1, 0, 0, 0, 4, 0]),
            (253402300799, [8099, 11, 31, 23, 59, 59, 5, 364]),
            (-62135596800, [-1899, 0, 1, 0, 0, 0, 1, 0]),
        ];
        for (time, [year, mon, mday, hour, min, sec, wday, yday]) in cases {
            ctx.write_u64(timer, time as u64);
            ctx.write_bytes(out, &[0xAA; GUEST_TM_SIZE]);
            ctx.set(Rdi, timer);
            ctx.set(Rsi, out);
            assert_eq!(cb_gmtime_s(&mut ctx), out as HleResult, "time {time}");
            let b = ctx.read_bytes(out, GUEST_TM_SIZE).unwrap();
            let f = |i: usize| i32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap());
            assert_eq!([f(5), f(4), f(3), f(2), f(1), f(0), f(6), f(7), f(8)], [year, mon, mday, hour, min, sec, wday, yday, 0], "time {time}");
            assert_eq!(&b[36..40], &[0; 4], "padding after tm_isdst");
            assert_eq!(i64::from_le_bytes(b[TM_GMTOFF..TM_GMTOFF + 8].try_into().unwrap()), 0);
            let zone = u64::from_le_bytes(b[TM_ZONE..TM_ZONE + 8].try_into().unwrap());
            assert_eq!(ctx.read_cstr_bytes(zone, 8).unwrap(), b"UTC");
        }
        // Years beyond an int are refused.
        ctx.write_u64(timer, i64::MIN as u64);
        assert_eq!(cb_gmtime_s(&mut ctx), 0);
        // strftime reads the guest layout.
        ctx.write_u64(timer, 0);
        assert_eq!(cb_gmtime_s(&mut ctx), out as HleResult);
        ctx.write_bytes(BASE + 0x200, b"%Y-%m-%d %H:%M:%S %Z\0");
        for (r, v) in [(Rdi, BASE + 0x300), (Rsi, 64), (Rdx, BASE + 0x200), (crate::context::CpuRegister::Rcx, out)] {
            ctx.set(r, v);
        }
        let zone = host_zone_name(false);
        let mut expected = b"1970-01-01 00:00:00 ".to_vec();
        expected.extend_from_slice(&zone);
        assert_eq!(cb_strftime(&mut ctx) as usize, expected.len());
        assert_eq!(ctx.read_cstr_bytes(BASE + 0x300, 64).unwrap(), expected);
        vm.unmap(BASE, 0x1000).unwrap();
    }

    /// strftime formats the conversions the console has (behaviour
    /// measured on hardware): clamped fields, week numbers, the zone from
    /// the zone state rather than the tm, unknown conversions and the
    /// FreeBSD-only ones copied through, and a buffer without room for
    /// the whole text and the terminator failing.
    #[test]
    fn strftime_formats_console_conversions() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x7C45_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let (timer, guest, fmt, out) = (BASE, BASE + 0x40, BASE + 0x100, BASE + 0x200);
        ctx.write_u64(timer, 1_700_000_000);
        ctx.set(Rdi, timer);
        ctx.set(Rsi, guest);
        assert_eq!(cb_gmtime_s(&mut ctx), guest as HleResult);
        let format = |ctx: &mut CpuContext, f: &str, size: u64| -> (HleResult, Vec<u8>) {
            ctx.write_bytes(fmt, &[f.as_bytes(), b"\0"].concat());
            ctx.write_bytes(out, &[b'x'; 96]);
            for (r, v) in [(Rdi, out), (Rsi, size), (Rdx, fmt), (crate::context::CpuRegister::Rcx, guest)] {
                ctx.set(r, v);
            }
            let n = cb_strftime(ctx);
            (n, ctx.read_cstr_bytes(out, 96).unwrap())
        };
        let check = |ctx: &mut CpuContext, f: &str, expected: &str| {
            let (n, text) = format(ctx, f, 96);
            assert_eq!((n as usize, String::from_utf8(text).unwrap()), (expected.len(), expected.to_string()), "format {f:?}");
        };
        // 2023-11-14 22:13:20, Tuesday, day 318 of the year.
        check(&mut ctx, "%Y-%m-%d %H:%M:%S", "2023-11-14 22:13:20");
        check(&mut ctx, "%a|%A|%b|%B", "Tue|Tuesday|Nov|November");
        check(&mut ctx, "%c", "Tue Nov 14 22:13:20 2023");
        check(&mut ctx, "%C|%y|%j|%u|%w", "20|23|318|2|2");
        check(&mut ctx, "%U|%W|%V|%G|%g", "46|46|46|2023|23");
        check(&mut ctx, "%D|%x|%F|%r|%R|%T|%X", "11/14/23|11/14/23|2023-11-14|10:13:20 PM|22:13|22:13:20|22:13:20");
        check(&mut ctx, "%p|%I", "PM|10");
        // Conversions the console does not have are copied through.
        check(&mut ctx, "%k|%l|%s|%v|%+|%q", "%k|%l|%s|%v|%+|%q");
        check(&mut ctx, "%-d|%_d|%0d", "%-d|%_d|%0d");
        check(&mut ctx, "%%%d", "%14");
        // The E and O modifiers are stepped over; a format ending after
        // one fails.
        check(&mut ctx, "%Od|%EY", "14|2023");
        assert_eq!(format(&mut ctx, "d%E", 96).0, 0);
        // %z formats the tm's own gmtoff plus the daylight hour for a
        // positive isdst; the host zone is not consulted (measured on
        // hardware). %Z names the zone state, not the tm.
        check(&mut ctx, "%z", "+0000");
        ctx.write_u32(guest + 32, 1); // tm_isdst east
        check(&mut ctx, "%z", "+0100");
        ctx.write_u32(guest + 32, -1_i32 as u32); // unknown daylight
        check(&mut ctx, "%z", "+0000");
        ctx.write_u64(guest + 40, (-3i64 * 3600) as u64);
        ctx.write_u32(guest + 32, 0);
        check(&mut ctx, "%z", "-0300");
        ctx.write_u32(guest + 32, 1);
        check(&mut ctx, "%z", "-0200");
        ctx.write_u32(guest + 32, 0);
        ctx.write_u64(guest + 40, 0);
        let name = host_zone_name(false);
        check(&mut ctx, "%Z", &String::from_utf8(name).unwrap());
        // Fields wrap at their width, negative ones read 0, out-of-range
        // names read the first or the empty name.
        ctx.write_u32(guest + 12, 61);
        check(&mut ctx, "%d|%e", "61|61");
        ctx.write_u32(guest + 12, 0xFFFF_FFFF);
        check(&mut ctx, "%d", "00");
        ctx.write_u32(guest + 12, 0);
        ctx.write_u32(guest + 8, 0xFFFF_FFFF);
        check(&mut ctx, "%H", "00");
        ctx.write_u32(guest + 8, 22);
        ctx.write_u32(guest + 24, -1_i32 as u32); // tm_wday
        check(&mut ctx, "%a|%A|%w|%u", "Sun|Sun|0|0");
        ctx.write_u32(guest + 24, 7);
        check(&mut ctx, "%a|%A", "|");
        ctx.write_u32(guest + 24, 2);
        ctx.write_u32(guest + 16, 12); // tm_mon
        check(&mut ctx, "%b|%B", "|");
        ctx.write_u32(guest + 16, -2_i32 as u32);
        check(&mut ctx, "%b|%B|%m", "Jan|Jan|00");
        ctx.write_u32(guest + 16, 10);
        // An ISO week crossing the year boundary.
        ctx.write_u64(timer, 1_609_459_200);
        ctx.set(Rdi, timer);
        ctx.set(Rsi, guest);
        cb_gmtime_s(&mut ctx);
        check(&mut ctx, "%G-%V-%u", "2020-53-5");
        // A piece that does not fit drops and fails the call; an exact
        // fit still needs the terminator.
        for (f, len) in [("%H", 2u64), ("%H:%M", 5), ("%Y", 4)] {
            assert_eq!(format(&mut ctx, f, len + 1).0, len as HleResult, "exact fit {f}");
            assert_eq!(format(&mut ctx, f, len + 1).1.len(), len as usize);
            assert_eq!(format(&mut ctx, f, len).0, 0, "one byte short {f}");
        }
        assert_eq!(format(&mut ctx, "%H:%M", 4).0, 0, "an undersized buffer");
        vm.unmap(BASE, 0x1000).unwrap();
    }

    /// ctime is asctime of the local time; an unconvertible time gives the
    /// question-mark text and EINVAL.
    #[test]
    fn ctime_formats_like_asctime() {
        let epoch = utc_calendar(0).unwrap();
        assert_eq!(asctime_text(&epoch), "Thu Jan  1 00:00:00 1970\n");
        let leap = utc_calendar(951_782_400 + 13 * 3600 + 4 * 60 + 5).unwrap();
        assert_eq!(asctime_text(&leap), "Tue Feb 29 13:04:05 2000\n");
        std::thread::spawn(|| {
            let vm: &'static ps5vm::VirtualMemory = Box::leak(Box::new(ps5vm::VirtualMemory::new()));
            const BASE: u64 = 0x6A11_0000_0000;
            vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
            let mut ctx = CpuContext::new(vm);
            let errno = crate::kernel::bind_test_errno(&ctx);
            let text = |ctx: &CpuContext, addr: u64| String::from_utf8(ctx.read_cstr_bytes(addr, 64).unwrap()).unwrap();
            let t: i64 = 951_782_400 + 13 * 3600 + 4 * 60 + 5;
            ctx.write_u64(BASE, t as u64);
            ctx.set(Rdi, BASE);
            let out = cb_ctime(&mut ctx) as u64;
            let mut host: libc::tm = unsafe { std::mem::zeroed() };
            assert!(!unsafe { libc::localtime_r(&(t as libc::time_t), &mut host) }.is_null());
            assert_eq!(text(&ctx, out), asctime_text(&host));
            assert_eq!(ctx.read_u32(errno), Some(0));
            ctx.write_u64(BASE, i64::MAX as u64);
            let unknown = cb_ctime(&mut ctx) as u64;
            assert_eq!(text(&ctx, unknown), CTIME_UNKNOWN);
            assert_eq!(ctx.read_u32(errno), Some(EINVAL as u32));
        })
        .join()
        .unwrap();
    }

    /// gethostname copies a fixed guest name; short buffers get the prefix
    /// and ENAMETOOLONG.
    #[test]
    fn gethostname_is_fixed_and_bounded() {
        assert_eq!(GUEST_HOSTNAME, b"\0", "the console's host name is empty");
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5F85_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let name = &GUEST_HOSTNAME[..GUEST_HOSTNAME.len() - 1];
        for len in 0..=name.len() + 2 {
            ctx.write_bytes(BASE, &[b'#'; 16]);
            ctx.set(Rdi, BASE + 1);
            ctx.set(Rsi, len as u64);
            let r = cb_gethostname(&mut ctx);
            let buf = ctx.read_bytes(BASE, 16).unwrap();
            assert_eq!(buf[0], b'#');
            if len <= name.len() {
                assert_eq!(r, -1, "len {len}");
                assert_eq!(&buf[1..1 + len], &name[..len]);
                assert_eq!(buf[1 + len], b'#');
            } else {
                assert_eq!(r, 0, "len {len}");
                assert_eq!(&buf[1..2 + name.len()], GUEST_HOSTNAME);
                assert!(buf[2 + name.len()..].iter().all(|&b| b == b'#'));
            }
        }
        ctx.set(Rsi, u64::MAX);
        assert_eq!(cb_gethostname(&mut ctx), 0);
        ctx.set(Rdi, 0);
        assert_eq!(cb_gethostname(&mut ctx), 0, "a NULL name");
        vm.unmap(BASE, 0x1000).unwrap();
    }

    #[test]
    fn memcmp_compares_bytes_and_fills_unreadable_with_zero() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5C28_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.write_bytes(BASE, b"hello");
        ctx.write_bytes(BASE + 0x10, b"help");
        ctx.set(Rdi, BASE);
        ctx.set(Rsi, BASE + 0x10);
        ctx.set(Rdx, 5);
        // "hello" vs "help": 'l' (0x6C) > 'p' (0x70) is false -> negative.
        assert!(cb_memcmp(&mut ctx) < 0);
        // Equal ranges compare 0.
        ctx.set(Rsi, BASE);
        assert_eq!(cb_memcmp(&mut ctx), 0);
        // An unreadable side reads as zeros: "hello" > "\0\0...".
        ctx.set(Rsi, BASE + 0x2000); // outside the mapping
        assert!(cb_memcmp(&mut ctx) > 0);
        // Zero length is always 0.
        ctx.set(Rdx, 0);
        assert_eq!(cb_memcmp(&mut ctx), 0);
    }

    #[test]
    fn libc_fallback_string_functions() {
        use crate::context::CpuRegister::Rcx;
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5D55_0000_0000;
        vm.map(BASE, 0x2000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let (a, b, d) = (BASE, BASE + 0x100, BASE + 0x200);
        ctx.write_bytes(a, b"hello world\0");
        ctx.write_bytes(b, b"wor\0");
        let set = |ctx: &mut CpuContext, v: &[u64]| {
            for (r, x) in [Rdi, Rsi, Rdx, Rcx].iter().zip(v) {
                ctx.set(*r, *x);
            }
        };
        set(&mut ctx, &[a, b'o' as u64]);
        assert_eq!(cb_strchr(&mut ctx), (a + 4) as HleResult);
        assert_eq!(cb_strrchr(&mut ctx), (a + 7) as HleResult);
        set(&mut ctx, &[a, 0]);
        assert_eq!(cb_strchr(&mut ctx), (a + 11) as HleResult);
        set(&mut ctx, &[a, b'z' as u64]);
        assert_eq!(cb_strchr(&mut ctx), 0);
        set(&mut ctx, &[a, b]);
        assert_eq!(cb_strstr(&mut ctx), (a + 6) as HleResult);
        set(&mut ctx, &[a, a + 11]);
        assert_eq!(cb_strstr(&mut ctx), a as HleResult);

        // strncpy zero-fills; strcat appends.
        ctx.write_bytes(d, &[0xAA; 8]);
        set(&mut ctx, &[d, b, 6]);
        assert_eq!(cb_strncpy(&mut ctx), d as HleResult);
        assert_eq!(ctx.read_bytes(d, 7).unwrap(), b"wor\0\0\0\xAA");
        set(&mut ctx, &[d, b]);
        assert_eq!(cb_strcat(&mut ctx), d as HleResult);
        assert_eq!(ctx.read_cstr_bytes(d, 16).unwrap(), b"worwor");

        // strcpy_s(dest, destsz, src): errno results, dest emptied on error.
        set(&mut ctx, &[d, 3, b]);
        assert_eq!(cb_strcpy_s(&mut ctx), ERANGE as HleResult);
        assert_eq!(ctx.read_bytes(d, 1).unwrap(), [0]);
        set(&mut ctx, &[d, 4, b]);
        assert_eq!(cb_strcpy_s(&mut ctx), 0);
        assert_eq!(ctx.read_cstr_bytes(d, 16).unwrap(), b"wor");
        set(&mut ctx, &[d, 4, 0]);
        assert_eq!(cb_strcpy_s(&mut ctx), EINVAL as HleResult);

        // strtol / strtoul / atoi with end pointers.
        ctx.write_bytes(a, b"  -0x1Fz\0");
        set(&mut ctx, &[a, d, 16]);
        assert_eq!(cb_strtol(&mut ctx), -31);
        assert_eq!(ctx.read_u64(d), Some(a + 7));
        ctx.write_bytes(a, b"777 \0");
        set(&mut ctx, &[a, 0, 8]);
        assert_eq!(cb_strtoul(&mut ctx), 0o777);
        set(&mut ctx, &[a]);
        assert_eq!(cb_atoi(&mut ctx), 777);
    }

    #[test]
    fn libc_snprintf_truncates_and_counts() {
        use crate::context::CpuRegister::{Rcx, Rsp};
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5D56_0000_0000;
        vm.map(BASE, 0x2000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let (fmt, dst) = (BASE, BASE + 0x100);
        ctx.write_bytes(fmt, b"%d-%s\0");
        ctx.write_bytes(BASE + 0x80, b"abc\0");
        ctx.write_bytes(dst, &[0xAA; 16]);
        ctx.set(Rsp, BASE + 0x1000);
        for (r, v) in [(Rdi, dst), (Rsi, 5), (Rdx, fmt), (Rcx, 1234)] {
            ctx.set(r, v);
        }
        ctx.set(crate::context::CpuRegister::R8, BASE + 0x80);
        // The full length is returned; at most size-1 bytes plus NUL are stored.
        assert_eq!(cb_snprintf(&mut ctx), 8);
        assert_eq!(ctx.read_bytes(dst, 6).unwrap(), b"1234\0\xAA");
        ctx.set(Rsi, 0);
        ctx.write_bytes(dst, &[0xAA; 2]);
        assert_eq!(cb_snprintf(&mut ctx), 8);
        assert_eq!(ctx.read_bytes(dst, 1).unwrap(), [0xAA]);
    }
}
