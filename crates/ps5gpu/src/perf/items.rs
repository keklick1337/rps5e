// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Recording CPU per program (`RPS5E_ITEM_CPU=1`, diagnostics): the
//! thread CPU time each draw (keyed by its pixel program, or its vertex
//! program without one) and each dispatch (its compute program) takes on
//! the recording thread, summed per program and logged as the top
//! programs every ten seconds (`[item-cpu]`). Shows which programs a
//! frame's recording time goes to when the per-function profile is flat.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Kinds of work told apart in the keys.
pub const DRAW: u64 = 1;
pub const DISPATCH: u64 = 2;

pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("RPS5E_ITEM_CPU").is_ok_and(|v| v == "1"))
}

const SPANS: usize = crate::backend::profile::NAMES.len();

struct Table {
    /// (kind << 56 | program) → (items, nanoseconds, profile spans).
    sums: HashMap<u64, (u64, u64, Box<[u64; SPANS]>)>,
    last: std::time::Instant,
}

std::thread_local! {
    /// The calling thread's profile spans (`RPS5E_GPU_PROFILE`) so far:
    /// an item's spans are the difference across it.
    static SPAN_SUMS: [std::cell::Cell<u64>; SPANS] = [const { std::cell::Cell::new(0) }; SPANS];
}

/// A profile span of `ns` ended on this thread.
pub fn note_span(k: usize, ns: u64) {
    let _ = SPAN_SUMS.try_with(|s| s[k].set(s[k].get() + ns));
}

fn spans() -> [u64; SPANS] {
    SPAN_SUMS.try_with(|s| std::array::from_fn(|i| s[i].get())).unwrap_or([0; SPANS])
}

static TABLE: Mutex<Option<Table>> = Mutex::new(None);

/// Run `f` (one draw or dispatch) and charge its thread CPU time to
/// `program`.
pub fn measure<R>(kind: u64, program: u64, f: impl FnOnce() -> R) -> R {
    if !enabled() {
        return f();
    }
    let s0 = spans();
    let t0 = crate::backend::profile::thread_cpu();
    let r = f();
    let ns = crate::backend::profile::thread_cpu().saturating_sub(t0);
    let s1 = spans();
    let mut g = TABLE.lock().unwrap_or_else(|e| e.into_inner());
    let t = g.get_or_insert_with(|| Table { sums: HashMap::new(), last: std::time::Instant::now() });
    let e = t.sums.entry(kind << 56 | (program & ((1 << 56) - 1))).or_insert_with(|| (0, 0, Box::new([0; SPANS])));
    e.0 += 1;
    e.1 += ns;
    for i in 0..SPANS {
        e.2[i] += s1[i] - s0[i];
    }
    if t.last.elapsed() >= std::time::Duration::from_secs(10) {
        t.last = std::time::Instant::now();
        let total: u64 = t.sums.values().map(|v| v.1).sum();
        let items: u64 = t.sums.values().map(|v| v.0).sum();
        let mut v: Vec<_> = t.sums.iter().collect();
        v.sort_by(|a, b| b.1 .1.cmp(&a.1 .1));
        let mut line = format!("[item-cpu] {} items {} ms:", items, total / 1_000_000);
        for (k, (n, ns, sp)) in v.iter().take(16) {
            let kind = if *k >> 56 == DRAW { 'd' } else { 'c' };
            line.push_str(&format!(" {kind}{:#x}={}x/{:.1}us/{:.1}%", *k & ((1 << 56) - 1), n, *ns as f64 / *n as f64 / 1000.0, *ns as f64 * 100.0 / total.max(1) as f64));
            // The largest spans per item (wall time, nested spans overlap).
            let mut top: Vec<(usize, u64)> = sp.iter().copied().enumerate().filter(|&(i, v)| v > 0 && i != 0 && i != 1 && i != 33).collect();
            top.sort_by(|a, b| b.1.cmp(&a.1));
            let names: Vec<String> = top.iter().take(5).map(|&(i, v)| format!("{}:{:.1}", crate::backend::profile::NAMES[i], v as f64 / *n as f64 / 1000.0)).collect();
            line.push_str(&format!("[{}]", names.join(",")));
        }
        log::info!("{line}");
        t.sums.clear();
    }
    r
}
