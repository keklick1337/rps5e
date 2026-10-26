// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Positioned events: guest ranges that command-processor work queued
//! behind earlier work will write, with the bytes it writes when those are
//! known (a fill, a CPU write; None for a copy). The interpreter's own CPU
//! accesses over them are ordered after them: a write is queued behind
//! them, a read sees the queued bytes over guest memory (or waits for a
//! copy).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Default)]
pub(crate) struct PendingEvents {
    count: AtomicU64,
    serial: AtomicU64,
    /// In queue order (the recorder removes them from the front).
    entries: Mutex<VecDeque<PendingEvent>>,
}

#[derive(Clone)]
struct PendingEvent {
    serial: u64,
    addr: u64,
    len: u64,
    data: Option<Arc<EventData>>,
}

pub(crate) enum EventData {
    Pattern(u32),
    Bytes(Vec<u8>),
}

impl EventData {
    /// The byte this event writes at offset `o` of its range.
    fn byte(&self, o: u64) -> u8 {
        match self {
            EventData::Pattern(v) => v.to_le_bytes()[(o % 4) as usize],
            EventData::Bytes(b) => b.get(o as usize).copied().unwrap_or(0),
        }
    }
}

impl PendingEvents {
    /// A queued write of `[addr, addr+len)` (`data`: what it writes, when
    /// known). Returns its serial for [`Self::remove`].
    pub(crate) fn add(&self, addr: u64, len: u64, data: Option<EventData>) -> u64 {
        let serial = self.serial.fetch_add(1, Ordering::AcqRel);
        self.entries.lock().unwrap_or_else(|e| e.into_inner()).push_back(PendingEvent { serial, addr, len, data: data.map(Arc::new) });
        self.count.fetch_add(1, Ordering::AcqRel);
        serial
    }

    /// The write happened.
    pub(crate) fn remove(&self, serial: u64) {
        let mut r = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        if r.front().is_some_and(|e| e.serial == serial) {
            r.pop_front();
        } else if let Some(i) = r.iter().position(|e| e.serial == serial) {
            r.remove(i);
        } else {
            return;
        }
        self.count.fetch_sub(1, Ordering::AcqRel);
    }

    /// Forget every queued write (they were dropped unexecuted).
    pub(crate) fn clear(&self) {
        let mut r = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        self.count.fetch_sub(r.len() as u64, Ordering::AcqRel);
        r.clear();
    }

    fn overlapping(&self, addr: u64, len: u64) -> Vec<PendingEvent> {
        if self.count.load(Ordering::Acquire) == 0 {
            return Vec::new();
        }
        let end = addr + len.max(1);
        self.entries.lock().unwrap_or_else(|e| e.into_inner()).iter().filter(|e| e.addr < end && addr < e.addr + e.len).cloned().collect()
    }

    pub(crate) fn overlaps(&self, addr: u64, len: u64) -> bool {
        if self.count.load(Ordering::Acquire) == 0 {
            return false;
        }
        let end = addr + len.max(1);
        self.entries.lock().unwrap_or_else(|e| e.into_inner()).iter().any(|e| e.addr < end && addr < e.addr + e.len)
    }

    /// `read` (guest memory) with the queued writes over the range applied
    /// in queue order; None when one of them is a copy (bytes unknown).
    pub(crate) fn compose(&self, addr: u64, out: &mut [u8], read: &dyn Fn(&mut [u8]) -> bool) -> Option<bool> {
        // Taken before memory is read: a write that runs meanwhile writes
        // the same bytes again.
        let pending = self.overlapping(addr, out.len() as u64);
        if pending.iter().any(|t| t.data.is_none()) {
            return None;
        }
        if !read(out) {
            return Some(false);
        }
        for t in &pending {
            let data = t.data.as_ref()?;
            for (i, b) in out.iter_mut().enumerate() {
                let a = addr + i as u64;
                if a >= t.addr && a < t.addr + t.len {
                    *b = data.byte(a - t.addr);
                }
            }
        }
        Some(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_see_queued_fills_and_writes_in_order() {
        let t = PendingEvents::default();
        let memory = |o: &mut [u8]| {
            o.fill(0xAA);
            true
        };
        let mut out = [0u8; 8];
        assert_eq!(t.compose(0x1000, &mut out, &memory), Some(true));
        assert_eq!(out, [0xAA; 8]);
        let fill = t.add(0x1000, 8, Some(EventData::Pattern(0x0403_0201)));
        let write = t.add(0x1002, 2, Some(EventData::Bytes(vec![9, 8])));
        assert_eq!(t.compose(0x0FFE, &mut out, &memory), Some(true));
        assert_eq!(out, [0xAA, 0xAA, 1, 2, 9, 8, 1, 2]);
        t.remove(fill);
        t.remove(write);
        assert!(!t.overlaps(0x1000, 8));
        t.add(0x1004, 4, None);
        assert_eq!(t.compose(0x1000, &mut out, &memory), None);
        t.clear();
        assert!(!t.overlaps(0x1000, 8));
    }
}
