// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Interval bookkeeping for the memory managers: how often each byte of a
//! space is referenced ([`RefSpans`]), and a map of spans that may overlap
//! each other with a cheap overlap search ([`SpanIndex`]). Both cost a
//! tree lookup plus the spans a call touches, whatever else they hold.

use std::collections::{BTreeMap, BTreeSet};

/// Reference counts over a space, as disjoint spans `start -> (end,
/// count)`; neighbours with the same count are one span.
#[derive(Default, Debug)]
pub(crate) struct RefSpans {
    spans: BTreeMap<u64, (u64, u32)>,
}

impl RefSpans {
    /// Make `at` a span boundary.
    fn split(&mut self, at: u64) {
        if let Some((&s, &(e, n))) = self.spans.range(..at).next_back() {
            if e > at {
                self.spans.insert(s, (at, n));
                self.spans.insert(at, (e, n));
            }
        }
    }

    /// Join the spans around the boundary `at` when their counts agree.
    fn join(&mut self, at: u64) {
        let Some(&(e, n)) = self.spans.get(&at) else { return };
        if let Some((&ps, &(pe, pn))) = self.spans.range(..at).next_back() {
            if pe == at && pn == n {
                self.spans.remove(&at);
                self.spans.insert(ps, (e, n));
            }
        }
    }

    /// One more reference to every byte of `[start, start + len)`.
    pub fn add(&mut self, start: u64, len: u64) {
        let Some(end) = start.checked_add(len).filter(|_| len != 0) else { return };
        self.split(start);
        self.split(end);
        let mut at = start;
        while at < end {
            match self.spans.range(at..end).next().map(|(&s, &(e, _))| (s, e)) {
                Some((s, e)) if s == at => {
                    self.spans.get_mut(&s).unwrap().1 += 1;
                    at = e;
                }
                next => {
                    let stop = next.map_or(end, |(s, _)| s);
                    self.spans.insert(at, (stop, 1));
                    at = stop;
                }
            }
        }
        self.join_range(start, end);
    }

    /// One reference less to every referenced byte of `[start, start + len)`.
    pub fn sub(&mut self, start: u64, len: u64) {
        let Some(end) = start.checked_add(len).filter(|_| len != 0) else { return };
        self.split(start);
        self.split(end);
        let mut at = start;
        while let Some((s, e, n)) = self.spans.range(at..end).next().map(|(&s, &(e, n))| (s, e, n)) {
            at = e;
            if n <= 1 {
                self.spans.remove(&s);
            } else {
                self.spans.insert(s, (e, n - 1));
            }
        }
        self.join_range(start, end);
    }

    fn join_range(&mut self, start: u64, end: u64) {
        let mut inner = start;
        while inner <= end {
            let Some((&s, _)) = self.spans.range(inner..=end).next() else { break };
            self.join(s);
            match s.checked_add(1) {
                Some(n) => inner = n,
                None => break,
            }
        }
    }

    /// The referenced parts of `[start, start + len)`: `(start, length)` in
    /// address order.
    pub fn referenced(&self, start: u64, len: u64) -> impl Iterator<Item = (u64, u64)> + '_ {
        let end = start.saturating_add(len);
        let first = match self.spans.range(..=start).next_back() {
            Some((&s, &(e, _))) if e > start => s,
            _ => start,
        };
        self.spans.range(first..end).map(move |(&s, &(e, _))| (s.max(start), e.min(end) - s.max(start)))
    }

    /// The parts of `[start, start + len)` nothing references, in address
    /// order.
    pub fn unreferenced(&self, start: u64, len: u64, mut f: impl FnMut(u64, u64)) {
        let end = start.saturating_add(len);
        let mut at = start;
        for (s, l) in self.referenced(start, len) {
            if s > at {
                f(at, s - at);
            }
            at = s + l;
        }
        if at < end {
            f(at, end - at);
        }
    }

    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }
}

const CLASSES: usize = 64;

/// Spans `start -> (length, value)` with unique starts that may overlap
/// each other, with their starts also kept by length class (lengths of
/// `[2^k, 2^(k+1))`): a span of class `k` overlapping an address starts
/// less than `2^(k+1)` below it, so an overlap search looks at a bounded
/// stretch of every class in use instead of at every span.
#[derive(Debug)]
pub(crate) struct SpanIndex<V> {
    spans: BTreeMap<u64, (u64, V)>,
    classes: [BTreeSet<u64>; CLASSES],
    /// Bit `k`: class `k` holds a span.
    used: u64,
}

impl<V> Default for SpanIndex<V> {
    fn default() -> Self {
        Self { spans: BTreeMap::new(), classes: std::array::from_fn(|_| BTreeSet::new()), used: 0 }
    }
}

fn class_of(len: u64) -> usize {
    63 - len.max(1).leading_zeros() as usize
}

impl<V: Copy> SpanIndex<V> {
    /// Enter (or replace) the span at `start`; the replaced one returns.
    pub fn insert(&mut self, start: u64, (len, value): (u64, V)) -> Option<(u64, V)> {
        let old = self.remove(&start);
        self.spans.insert(start, (len, value));
        let k = class_of(len);
        self.classes[k].insert(start);
        self.used |= 1 << k;
        old
    }

    pub fn remove(&mut self, start: &u64) -> Option<(u64, V)> {
        let old = self.spans.remove(start)?;
        let k = class_of(old.0);
        self.classes[k].remove(start);
        if self.classes[k].is_empty() {
            self.used &= !(1 << k);
        }
        Some(old)
    }

    #[cfg(test)]
    pub fn get(&self, start: &u64) -> Option<&(u64, V)> {
        self.spans.get(start)
    }

    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    pub fn values(&self) -> impl Iterator<Item = &(u64, V)> {
        self.spans.values()
    }

    /// Starts of the spans that start in `[lo, hi)`.
    pub fn starts_in(&self, lo: u64, hi: u64) -> Vec<u64> {
        self.spans.range(lo..hi).map(|(&s, _)| s).collect()
    }

    /// The spans overlapping `[start, start + len)`: `(start, length,
    /// value)` by start.
    pub fn overlapping(&self, start: u64, len: u64, out: &mut Vec<(u64, u64, V)>) {
        out.clear();
        let end = start.saturating_add(len);
        let mut used = self.used;
        while used != 0 {
            let k = used.trailing_zeros() as usize;
            used &= used - 1;
            let reach = if k == 63 { u64::MAX } else { (2u64 << k) - 1 };
            for &s in self.classes[k].range(start.saturating_sub(reach)..end) {
                let (l, v) = self.spans[&s];
                if s.saturating_add(l) > start {
                    out.push((s, l, v));
                }
            }
        }
        out.sort_unstable_by_key(|e| e.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_counts_split_and_join() {
        let mut r = RefSpans::default();
        r.add(0x1000, 0x3000);
        r.add(0x2000, 0x4000);
        assert_eq!(r.referenced(0, 0x10000).collect::<Vec<_>>(), vec![(0x1000, 0x1000), (0x2000, 0x2000), (0x4000, 0x2000)]);
        let mut holes = Vec::new();
        r.unreferenced(0, 0x8000, |s, l| holes.push((s, l)));
        assert_eq!(holes, vec![(0, 0x1000), (0x6000, 0x2000)]);
        r.sub(0x1000, 0x3000);
        assert_eq!(r.referenced(0, 0x10000).collect::<Vec<_>>(), vec![(0x2000, 0x4000)], "equal neighbours are one span");
        assert_eq!(r.referenced(0x3000, 0x1000).collect::<Vec<_>>(), vec![(0x3000, 0x1000)]);
        r.sub(0x2000, 0x4000);
        assert!(r.is_empty());
        r.sub(0x2000, 0x4000);
        assert!(r.is_empty(), "nothing to release");
    }

    #[test]
    fn overlapping_spans_are_found_whatever_their_length() {
        let mut m: SpanIndex<u32> = SpanIndex::default();
        m.insert(0x1000, (0x10_0000, 1));
        m.insert(0x8000, (0x1000, 2));
        m.insert(0x9000, (0x1000, 3));
        m.insert(0x20_0000, (0x4000, 4));
        let mut out = Vec::new();
        m.overlapping(0x8800, 0x1000, &mut out);
        assert_eq!(out, vec![(0x1000, 0x10_0000, 1), (0x8000, 0x1000, 2), (0x9000, 0x1000, 3)]);
        m.overlapping(0x10_1000, 0xF_F000, &mut out);
        assert!(out.is_empty());
        m.overlapping(0x10_0000, 0x10_0001, &mut out);
        assert_eq!(out, vec![(0x1000, 0x10_0000, 1), (0x20_0000, 0x4000, 4)]);
        assert_eq!(m.insert(0x1000, (0x2000, 5)), Some((0x10_0000, 1)));
        m.overlapping(0x8800, 0x1000, &mut out);
        assert_eq!(out, vec![(0x8000, 0x1000, 2), (0x9000, 0x1000, 3)]);
        assert_eq!(m.remove(&0x8000), Some((0x1000, 2)));
        assert_eq!(m.remove(&0x8000), None);
    }
}
