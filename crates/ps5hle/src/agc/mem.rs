// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Raw guest-memory access shared by the AGC packet builders and the PM4
//! interpreter.
//!
//! Guest addresses are host addresses (native execution); every access is
//! validated against the guest mapping registry (`ps5vm::ranges`), so a bad
//! pointer fails cleanly instead of faulting. These helpers do not need a
//! `CpuContext`, which lets the GPU command thread use them too.

/// Recently used guest ranges of this thread: (registry generation, start,
/// end, readable, writable). Builders and the command processor touch the
/// same few mappings (command buffers, labels) over and over.
#[derive(Clone, Copy)]
struct CachedRange {
    generation: u64,
    start: u64,
    end: u64,
    read: bool,
    write: bool,
}

const RANGE_CACHE: usize = 4;

thread_local! {
    static RANGES: std::cell::Cell<([CachedRange; RANGE_CACHE], usize)> =
        const { std::cell::Cell::new(([CachedRange { generation: u64::MAX, start: 0, end: 0, read: false, write: false }; RANGE_CACHE], 0)) };
}

/// `ps5vm::ranges::accessible` with a per-thread cache of the ranges it
/// found (invalidated by any change to the registry).
pub fn accessible(addr: u64, len: u64, write: bool) -> bool {
    let Some(end) = addr.checked_add(len) else { return false };
    let generation = ps5vm::ranges::generation();
    let (mut cache, next) = RANGES.get();
    if cache.iter().any(|r| r.generation == generation && r.start <= addr && end <= r.end && if write { r.write } else { r.read }) {
        return true;
    }
    if !ps5vm::ranges::accessible(addr, len, write) {
        return false;
    }
    // Remember the range when one mapping covers the whole access.
    if let Some((start, range_end, prot)) = ps5vm::ranges::containing(addr) {
        if end <= range_end {
            cache[next] = CachedRange { generation, start, end: range_end, read: prot.read, write: prot.write };
            RANGES.set((cache, (next + 1) % RANGE_CACHE));
        }
    }
    true
}

/// Copy `buf.len()` bytes from guest `addr`.
pub fn read(addr: u64, buf: &mut [u8]) -> bool {
    if buf.is_empty() {
        return true;
    }
    if addr == 0 || !accessible(addr, buf.len() as u64, false) {
        return false;
    }
    // SAFETY: the range is a tracked, readable guest mapping.
    unsafe { std::ptr::copy_nonoverlapping(addr as *const u8, buf.as_mut_ptr(), buf.len()) };
    true
}

/// Copy `data` to guest `addr`.
pub fn write(addr: u64, data: &[u8]) -> bool {
    if data.is_empty() {
        return true;
    }
    if addr == 0 || !accessible(addr, data.len() as u64, true) {
        return false;
    }
    // SAFETY: the range is a tracked, writable guest mapping.
    unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), addr as *mut u8, data.len()) };
    true
}

pub fn r8(addr: u64) -> Option<u8> {
    let mut b = [0u8; 1];
    read(addr, &mut b).then_some(b[0])
}

pub fn r16(addr: u64) -> Option<u16> {
    let mut b = [0u8; 2];
    read(addr, &mut b).then(|| u16::from_le_bytes(b))
}

pub fn r32(addr: u64) -> Option<u32> {
    let mut b = [0u8; 4];
    read(addr, &mut b).then(|| u32::from_le_bytes(b))
}

pub fn r64(addr: u64) -> Option<u64> {
    let mut b = [0u8; 8];
    read(addr, &mut b).then(|| u64::from_le_bytes(b))
}

pub fn w8(addr: u64, v: u8) -> bool {
    write(addr, &[v])
}

pub fn w32(addr: u64, v: u32) -> bool {
    write(addr, &v.to_le_bytes())
}

pub fn w64(addr: u64, v: u64) -> bool {
    write(addr, &v.to_le_bytes())
}

/// Read `count` dwords.
pub fn read_dwords(addr: u64, count: usize) -> Option<Vec<u32>> {
    // Read straight into the words (little-endian host).
    let mut words = vec![0u32; count];
    // SAFETY: the u32 buffer viewed as its bytes; every bit pattern is a valid u32.
    let bytes = unsafe { std::slice::from_raw_parts_mut(words.as_mut_ptr() as *mut u8, count * 4) };
    read(addr, bytes).then_some(words)
}

/// [`read_dwords`] into `out` (reused; resized to `count`).
pub fn read_dwords_into(addr: u64, count: usize, out: &mut Vec<u32>) -> bool {
    out.clear();
    out.resize(count, 0);
    // SAFETY: the u32 buffer viewed as its bytes; every bit pattern is a valid u32.
    let bytes = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, count * 4) };
    read(addr, bytes)
}

/// Write a dword slice (no intermediate buffer).
pub fn write_dwords(addr: u64, words: &[u32]) -> bool {
    if words.is_empty() {
        return true;
    }
    let len = words.len() as u64 * 4;
    if addr == 0 || !accessible(addr, len, true) {
        return false;
    }
    // SAFETY: the range is a tracked, writable guest mapping.
    unsafe { std::ptr::copy_nonoverlapping(words.as_ptr() as *const u8, addr as *mut u8, len as usize) };
    true
}

/// [`write_dwords`] as the command processor publishes WRITE_DATA: every
/// dword is stored exactly once, in order, with one 32-bit store. A copy
/// routine may store some bytes twice (overlapping tail moves); a guest
/// that sees a completion label and resets it at once would then see the
/// label come back.
pub fn publish_dwords(addr: u64, words: &[u32]) -> bool {
    if words.is_empty() {
        return true;
    }
    let len = words.len() as u64 * 4;
    if addr == 0 || !accessible(addr, len, true) {
        return false;
    }
    let p = addr as *mut u32;
    for (i, &w) in words.iter().enumerate() {
        // SAFETY: the range is a tracked, writable guest mapping; an
        // unaligned destination is stored as a byte array (alignment 1).
        unsafe {
            if addr % 4 == 0 {
                std::ptr::write_volatile(p.add(i), w);
            } else {
                std::ptr::write_volatile(p.add(i) as *mut [u8; 4], w.to_le_bytes());
            }
        }
    }
    true
}

/// Fill `bytes` at guest `addr` with `value`.
pub fn fill(addr: u64, bytes: u64, value: u8) -> bool {
    if bytes == 0 {
        return true;
    }
    if addr == 0 || !accessible(addr, bytes, true) {
        return false;
    }
    // SAFETY: the range is a tracked, writable guest mapping.
    unsafe { std::ptr::write_bytes(addr as *mut u8, value, bytes as usize) };
    true
}

/// Fill `bytes` at guest `addr` with the little-endian dword `value`
/// repeated (a partial last dword gets its low bytes).
pub fn fill32(addr: u64, bytes: u64, value: u32) -> bool {
    if bytes == 0 {
        return true;
    }
    if addr == 0 || !accessible(addr, bytes, true) {
        return false;
    }
    let pattern = value.to_le_bytes();
    // SAFETY: the range is a tracked, writable guest mapping.
    let dst = unsafe { std::slice::from_raw_parts_mut(addr as *mut u8, bytes as usize) };
    if pattern.iter().all(|&b| b == pattern[0]) {
        dst.fill(pattern[0]);
    } else {
        for (i, b) in dst.iter_mut().enumerate() {
            *b = pattern[i & 3];
        }
    }
    true
}

/// Read a NUL-terminated string into `out` (at most `out.len() - 1` bytes,
/// the rest zeroed). Returns its length without the terminator; `None` when
/// unreadable or longer than that.
pub fn read_cstr_into(addr: u64, out: &mut [u8]) -> Option<usize> {
    let mut len = 0;
    while len < out.len() {
        // Up to the end of the current page in one checked read.
        let cur = addr + len as u64;
        let chunk = ((0x1000 - (cur & 0xFFF)) as usize).min(out.len() - len);
        if !read(cur, &mut out[len..len + chunk]) {
            return None;
        }
        if let Some(p) = out[len..len + chunk].iter().position(|&b| b == 0) {
            let end = len + p;
            out[end..].fill(0);
            return Some(end);
        }
        len += chunk;
    }
    None
}

/// Copy `bytes` from guest `src` to guest `dst` (ranges may overlap).
pub fn copy(dst: u64, src: u64, bytes: u64) -> bool {
    if bytes == 0 {
        return true;
    }
    if dst == 0 || src == 0 || !accessible(src, bytes, false) || !accessible(dst, bytes, true) {
        return false;
    }
    // SAFETY: both ranges are tracked guest mappings with the needed access.
    unsafe { std::ptr::copy(src as *const u8, dst as *mut u8, bytes as usize) };
    true
}

/// Read a NUL-terminated byte string of at most `max` bytes (without the
/// terminator). `None` when unreadable or unterminated.
pub fn read_cstr(addr: u64, max: usize) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    let mut cur = addr;
    while out.len() < max {
        // Read up to the end of the current page in one go.
        let page_left = (0x1000 - (cur & 0xFFF)) as usize;
        let chunk = page_left.min(max - out.len() + 1).max(1);
        let mut buf = vec![0u8; chunk];
        if !read(cur, &mut buf) {
            // Fall back to byte reads near a mapping edge.
            let b = r8(cur)?;
            if b == 0 {
                return Some(out);
            }
            out.push(b);
            cur += 1;
            continue;
        }
        if let Some(p) = buf.iter().position(|&b| b == 0) {
            out.extend_from_slice(&buf[..p]);
            return (out.len() <= max).then_some(out);
        }
        out.extend_from_slice(&buf);
        cur += chunk as u64;
    }
    None
}
