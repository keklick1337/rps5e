// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Hardware ray tracing for guest BVH queries (`RPS5E_HW_RT`).
//!
//! Guest ray-tracing shaders walk their BVHs in software: a loop of
//! `image_bvh_intersect_ray` box and triangle queries over nodes in guest
//! memory, with the any-hit work (alpha tests, masks, culling) in guest
//! code between the queries. The host mirrors each guest top-level BVH
//! (TLAS) as a Vulkan acceleration structure (one host BLAS per guest
//! BLAS, triangles taken from the guest's triangle nodes, instances from
//! the guest's instance records, the opaque bit and instance flags carried
//! over) and the translator gives each recognised traversal loop a copy
//! whose nodes come from host ray queries over the mirror
//! (`ps5shader::opt::rt_loops`, emitted by `shader::bvh::emit_rt`): the
//! guest's own instance, leaf-list and candidate code runs on what the RT
//! cores find, without the guest's box traversal.
//!
//! Upkeep ([`mirror`]): the mirrors update once per write generation of
//! the guest BVHs they copy ([`watch`]: the console's refits and builds
//! write them once a frame) and only for dispatches whose module reads
//! them; the triangle kernel compares every triangle with the last
//! update's and the host refits only the BLASes it found changed (read
//! once their update completed, without waits), builds them again when
//! their entry count or opacity changed; a host BLAS has room for a few
//! more entries than its guest BLAS. An instance whose BLAS the host lacks
//! (new, too large, stale) is a box for the traversal copies (the guest
//! walks it) and is built at the next update.
//!
//! Without a recognised loop, `RPS5E_HW_RT_PRUNE=1` keeps the earlier
//! scheme: a root query bounds the hit distances of each lane and box
//! queries drop the children outside them (the guest loop stays).

pub mod kernels;
pub mod layout;
pub mod mirror;

/// `RPS5E_HW_RT`: `0` off, `1` on (traversal loops on host ray queries
/// where the device has them), `2` on with logging and counters.
pub fn level() -> u32 {
    static LEVEL: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *LEVEL.get_or_init(|| std::env::var("RPS5E_HW_RT").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0))
}

/// The device should enable its ray-query extensions (always in this
/// crate's tests, which may turn the mirrors on).
pub fn requested() -> bool {
    level() != 0 || cfg!(test)
}

/// Words of the ray-tracing table. It lives in the fault buffer (bound
/// with the device-address page table to every stage that reads guest
/// memory through addresses, BVH queries among them) from word [`BASE`]
/// on; the indices below count from there.
pub mod table {
    /// First word of the table in the fault buffer (past the fault log).
    pub const BASE: u32 = 1024;
    /// Feedback counter (atomic, only grows).
    pub const FEEDBACK_COUNT: u32 = 0;
    /// Feedback ring: `FEEDBACK_SLOTS` pairs (base units low, high) of BVH
    /// roots queried while the host did not know them.
    pub const FEEDBACK: u32 = 4;
    pub const FEEDBACK_SLOTS: u32 = 32;
    /// Mirror entries: `ENTRY_COUNT` of `ENTRY_WORDS` words.
    pub const ENTRIES: u32 = FEEDBACK + 2 * FEEDBACK_SLOTS;
    pub const ENTRY_COUNT: u32 = 8;
    pub const ENTRY_WORDS: u32 = 8;
    /// Entry word 0: the guest BVH's base (T# units of 256 bytes, low 32
    /// bits); word 1: its high 8 bits and the flags below; words 2-3: the
    /// host TLAS's device address; words 4-5: device address of the set of
    /// guest BLAS bases its instances use (`SET_WORDS`-word slots, key =
    /// base + 1 as u64, 0 empty); word 6: slot count - 1 (a power of two).
    pub const WORD_BASE: u32 = 0;
    pub const WORD_FLAGS: u32 = 1;
    pub const WORD_AS: u32 = 2;
    pub const WORD_SET: u32 = 4;
    pub const WORD_SET_MASK: u32 = 6;
    /// Word 7: set by the shaders at a root query of the entry (the host
    /// clears it at every update and frees entries no shader uses).
    pub const WORD_USED: u32 = 7;
    /// The entry is in use.
    pub const KNOWN: u32 = 1 << 8;
    /// The entry mirrors a top-level BVH.
    pub const TLAS: u32 = 1 << 9;
    /// The host TLAS holds every instance of the guest TLAS (written by
    /// the GPU at each update).
    pub const VALID: u32 = 1 << 10;
    /// Every instance of the host TLAS has the host BLAS of its guest BLAS,
    /// current (none stands in as a box): a traversal may run on the host
    /// alone (cleared by the GPU at each update).
    pub const COMPLETE: u32 = 1 << 11;
    /// The guest BVH is a bottom level traced as the top (one identity
    /// instance).
    pub const SINGLE: u32 = 1 << 12;
    /// Feedback filter: a bit per hash of a BVH base; a root is reported
    /// only while its bit is clear (the host clears the filter now and
    /// then), so a wave reports an unknown BVH once, not at every query.
    pub const FILTER: u32 = ENTRIES + ENTRY_COUNT * ENTRY_WORDS;
    pub const FILTER_WORDS: u32 = 32;
    pub const SET_WORDS: u32 = 2;
    /// Probes of the BLAS set before a key counts as absent.
    pub const SET_PROBES: u32 = 16;
    /// Counters: of the pruning (`RPS5E_HW_RT=2` with
    /// `RPS5E_HW_RT_PRUNE=1`): root queries of a valid mirror (ray query
    /// made), of an invalid one, roots that dropped the state, ray queries
    /// without a hit, box queries, children kept; else of the instance
    /// kernels: instances standing in as boxes because their BLAS has no
    /// map slot, is too small, has no structure, is stale, is not of the
    /// format, and instances seen; then of the hardware traversals
    /// (`RPS5E_HW_RT=2`): traversals begun, candidates fetched, fetches past
    /// the candidate cap.
    pub const STATS: u32 = FILTER + FILTER_WORDS;
    pub const STAT_WORDS: u32 = 10;
    /// Device address (low, high) of the leaf-list table: per BLAS map
    /// slot, the device address of the BLAS's list starts (per leaf entry,
    /// the index of the first entry of its leaf list).
    pub const STARTS: u32 = STATS + STAT_WORDS;
    pub const WORDS: u32 = STARTS + 2;
    pub const BYTES: u64 = WORDS as u64 * 4;

    /// The set slot a base starts probing at.
    pub fn set_hash(base_lo: u32) -> u32 {
        base_lo.wrapping_mul(0x9E37_79B1) >> 12
    }
}

/// The node pointer of a BVH's root: the first BOX32 node after the
/// 256-byte header.
pub const ROOT_NODE_POINTER: u32 = 0x25;

/// Relative and absolute tolerance of the hit interval against box entry
/// and exit distances.
pub const T_RELATIVE: f32 = 1.0 / 1024.0;
pub const T_ABSOLUTE: f32 = 1.0 / 16384.0;

/// GPU writes to the guest BVHs the mirrors copy: the mirrors update only
/// when the generation moved since their last update (the console's own
/// refits and builds write the blocks once a frame).
pub mod watch {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::RwLock;

    static ON: AtomicBool = AtomicBool::new(false);
    static WRITES: AtomicU64 = AtomicU64::new(0);
    /// Watched byte ranges, sorted and disjoint.
    static RANGES: RwLock<Vec<(u64, u64)>> = RwLock::new(Vec::new());

    /// Watch `ranges` (start, end) from now on, replacing the set.
    pub fn set(mut ranges: Vec<(u64, u64)>) {
        ranges.sort_unstable();
        let mut merged: Vec<(u64, u64)> = Vec::with_capacity(ranges.len());
        for (a, b) in ranges {
            match merged.last_mut() {
                Some(last) if a <= last.1 => last.1 = last.1.max(b),
                _ => merged.push((a, b)),
            }
        }
        ON.store(!merged.is_empty(), Ordering::Relaxed);
        *RANGES.write().unwrap_or_else(|e| e.into_inner()) = merged;
        WRITES.fetch_add(1, Ordering::Relaxed);
    }

    /// `RPS5E_HW_RT=2`: per program, its writes that moved the generation
    /// (ranged, unranged).
    static TRIGGERS: std::sync::Mutex<Option<std::collections::HashMap<u64, (u64, u64)>>> = std::sync::Mutex::new(None);

    fn note(unranged: bool) {
        WRITES.fetch_add(1, Ordering::Relaxed);
        if super::level() >= 2 {
            let program = crate::backend::profile::preparing_program();
            let mut t = TRIGGERS.lock().unwrap_or_else(|e| e.into_inner());
            let e = t.get_or_insert_with(Default::default).entry(program).or_default();
            if unranged {
                e.1 += 1;
            } else {
                e.0 += 1;
            }
        }
    }

    /// The programs whose writes moved the generation since the last call
    /// (`RPS5E_HW_RT=2`), most first.
    pub fn take_triggers() -> Vec<(u64, u64, u64)> {
        let mut t = TRIGGERS.lock().unwrap_or_else(|e| e.into_inner());
        let mut v: Vec<(u64, u64, u64)> = t.take().unwrap_or_default().into_iter().map(|(p, (a, b))| (p, a, b)).collect();
        v.sort_by_key(|e| std::cmp::Reverse(e.1 + e.2));
        v
    }

    /// GPU work will write `[addr, addr + size)`.
    pub fn written(addr: u64, size: u64) {
        if !ON.load(Ordering::Relaxed) {
            return;
        }
        let r = RANGES.read().unwrap_or_else(|e| e.into_inner());
        let end = addr.saturating_add(size.max(1));
        let i = r.partition_point(|&(_, b)| b <= addr);
        if r.get(i).is_some_and(|&(a, _)| a < end) {
            drop(r);
            note(false);
        }
    }

    /// GPU work will write memory it names no ranges for (stores through
    /// device addresses).
    pub fn unranged() {
        if ON.load(Ordering::Relaxed) {
            note(true);
        }
    }

    pub fn generation() -> u64 {
        WRITES.load(Ordering::Relaxed)
    }
}
