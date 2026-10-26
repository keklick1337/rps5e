// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! [`ThroughSpans`]: the write-through spans of the buffer cache (guest
//! pages whose GPU data is being copied out), found by address.
//!
//! Every GPU-written buffer use cuts the spans its range overlaps, and
//! compute-heavy frames bind written buffers thousands of times while up
//! to [`ThroughSpans::MAX`] spans are remembered. Spans are kept by start
//! address with the largest span length, so a lookup visits only the spans
//! that start within that length before its range. Each span carries its
//! insertion number: overlapping spans answer newest first, and the oldest
//! half is dropped when the list is full. `RPS5E_THROUGH_INDEX=0` (or a
//! slow period of A/B scope `release`) answers every lookup by a scan of
//! all spans instead, with the same results.

use super::ThroughSpan;
use std::collections::BTreeMap;

#[derive(Default)]
pub(super) struct ThroughSpans {
    /// By (start, insertion number).
    map: BTreeMap<(u64, u64), ThroughSpan>,
    /// Insertion numbers handed out.
    seq: u64,
    /// No span is longer (recomputed when the list is trimmed).
    max_len: u64,
}

/// Lookups use the index now.
fn indexed() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::backend::ab::switch_on("RPS5E_THROUGH_INDEX")) && crate::backend::ab::release_fast("RPS5E_THROUGH_INDEX")
}

impl ThroughSpans {
    /// Spans remembered at most (older ones fall back to a copy).
    pub const MAX: usize = 1024;

    pub fn clear(&mut self) {
        self.map.clear();
        self.max_len = 0;
    }

    /// Remember a span (the newest). A full list drops its oldest half.
    pub fn push(&mut self, span: ThroughSpan) {
        if self.map.len() >= Self::MAX {
            self.trim(Self::MAX / 2);
        }
        self.seq += 1;
        self.max_len = self.max_len.max(span.end - span.start);
        self.map.insert((span.start, self.seq), span);
    }

    /// Drop the `n` oldest spans.
    fn trim(&mut self, n: usize) {
        let mut seqs: Vec<u64> = self.map.keys().map(|k| k.1).collect();
        if n >= seqs.len() {
            self.clear();
            return;
        }
        let (_, &mut cutoff, _) = seqs.select_nth_unstable(n);
        self.map.retain(|k, _| k.1 >= cutoff);
        self.max_len = self.map.values().map(|s| s.end - s.start).max().unwrap_or(0);
    }

    /// Keys of the spans overlapping `[start, end)`, in no particular order.
    fn overlapping_keys(&self, start: u64, end: u64) -> Vec<(u64, u64)> {
        let hit = |s: &ThroughSpan| s.start < end && start < s.end;
        if indexed() {
            let from = start.saturating_sub(self.max_len);
            self.map.range((from, 0)..(end, 0)).filter(|(_, s)| hit(s)).map(|(&k, _)| k).collect()
        } else {
            self.map.iter().filter(|(_, s)| hit(s)).map(|(&k, _)| k).collect()
        }
    }

    /// Whether a span overlaps `[start, end)`.
    pub fn overlaps(&self, start: u64, end: u64) -> bool {
        let hit = |s: &ThroughSpan| s.start < end && start < s.end;
        if indexed() {
            let from = start.saturating_sub(self.max_len);
            self.map.range((from, 0)..(end, 0)).any(|(_, s)| hit(s))
        } else {
            self.map.values().any(hit)
        }
    }

    /// The spans overlapping `[start, end)`, oldest first (the parts of
    /// one span, which never overlap each other, by address).
    pub fn overlapping_oldest_first(&self, start: u64, end: u64) -> Vec<&ThroughSpan> {
        let mut keys = self.overlapping_keys(start, end);
        keys.sort_unstable_by_key(|k| (k.1, k.0));
        keys.iter().map(|k| &self.map[k]).collect()
    }

    /// The newest span holding byte `addr`.
    pub fn newest_containing(&self, addr: u64) -> Option<&ThroughSpan> {
        let keys = self.overlapping_keys(addr, addr + 1);
        keys.iter().max_by_key(|k| k.1).map(|k| &self.map[k])
    }

    /// Cut `[start, end)` out of every span (the parts outside stay, with
    /// their age).
    pub fn cut(&mut self, start: u64, end: u64) {
        for k in self.overlapping_keys(start, end) {
            let sp = self.map.remove(&k).expect("listed span");
            if sp.start < start {
                self.map.insert((sp.start, k.1), sp.part(sp.start, start));
            }
            if sp.end > end {
                self.map.insert((end, k.1), sp.part(end, sp.end));
            }
        }
    }

    /// Drop the spans wholly inside `[start, end)`.
    pub fn remove_inside(&mut self, start: u64, end: u64) {
        for k in self.overlapping_keys(start, end) {
            if self.map[&k].start >= start && self.map[&k].end <= end {
                self.map.remove(&k);
            }
        }
    }

    #[cfg(test)]
    fn all(&self) -> Vec<(u64, u64, u64)> {
        let mut v: Vec<(u64, u64, u64)> = self.map.iter().map(|(k, s)| (k.1, s.start, s.end)).collect();
        v.sort_unstable();
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(start: u64, end: u64, tick: u64) -> ThroughSpan {
        ThroughSpan { start, end, tick, staged: None }
    }

    /// The indexed lookups answer as a list scanned in insertion order:
    /// random pushes, cuts, removals and lookups against a plain model.
    #[test]
    fn index_answers_as_an_ordered_list() {
        let mut spans = ThroughSpans::default();
        // The model: (insertion number, start, end, tick) in insertion order.
        let mut model: Vec<(u64, u64, u64, u64)> = Vec::new();
        let mut seq = 0u64;
        let mut rng = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let page = 0x1000u64;
        for step in 0..20_000u64 {
            let r = next();
            let a = (r % 4096) * page;
            let n = [1, 2, 16, 300, 2000][(r >> 20) as usize % 5] * page;
            match (r >> 40) % 8 {
                0..=3 => {
                    if model.len() >= ThroughSpans::MAX {
                        model.sort_by_key(|m| m.0);
                        model.drain(..ThroughSpans::MAX / 2);
                    }
                    seq += 1;
                    model.push((seq, a, a + n, step));
                    spans.push(span(a, a + n, step));
                }
                4 => {
                    let mut out = Vec::new();
                    for &(q, s, e, t) in &model {
                        if s < a + n && a < e {
                            if s < a {
                                out.push((q, s, a, t));
                            }
                            if e > a + n {
                                out.push((q, a + n, e, t));
                            }
                        } else {
                            out.push((q, s, e, t));
                        }
                    }
                    model = out;
                    spans.cut(a, a + n);
                }
                5 => {
                    model.retain(|&(_, s, e, _)| s < a || e > a + n);
                    spans.remove_inside(a, a + n);
                }
                6 => {
                    let want = model.iter().filter(|m| m.1 <= a && a < m.2).max_by_key(|m| m.0).map(|m| (m.1, m.2, m.3));
                    assert_eq!(spans.newest_containing(a).map(|s| (s.start, s.end, s.tick)), want, "newest at {a:#x}");
                }
                _ => {
                    let mut want: Vec<_> = model.iter().filter(|m| m.1 < a + n && a < m.2).copied().collect();
                    want.sort_by_key(|m| (m.0, m.1));
                    let got: Vec<_> = spans.overlapping_oldest_first(a, a + n).iter().map(|s| (s.start, s.end, s.tick)).collect();
                    assert_eq!(got, want.iter().map(|m| (m.1, m.2, m.3)).collect::<Vec<_>>(), "overlapping {a:#x}+{n:#x}");
                    assert_eq!(spans.overlaps(a, a + n), !want.is_empty());
                }
            }
            if step % 1000 == 0 {
                let mut m: Vec<_> = model.iter().map(|m| (m.0, m.1, m.2)).collect();
                m.sort_unstable();
                assert_eq!(spans.all(), m, "step {step}");
            }
        }
    }
}
