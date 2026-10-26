// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! One bit per page of a 4 MiB tracking block.

use super::PAGES_PER_BLOCK;

const WORDS: usize = PAGES_PER_BLOCK / 64;

#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct PageMask {
    words: [u64; WORDS],
}

impl std::fmt::Debug for PageMask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let runs: Vec<(usize, usize)> = self.runs().collect();
        write!(f, "PageMask{runs:?}")
    }
}

fn range_word(word: usize, start: usize, end: usize) -> u64 {
    if end > PAGES_PER_BLOCK {
        return 0;
    }
    let low = start.max(word * 64).saturating_sub(word * 64);
    let high = end.min((word + 1) * 64).saturating_sub(word * 64);
    if low >= high {
        return 0;
    }
    let hi_mask = if high == 64 { u64::MAX } else { (1u64 << high) - 1 };
    hi_mask & (u64::MAX << low)
}

impl PageMask {
    pub const fn empty() -> Self {
        Self { words: [0; WORDS] }
    }

    pub const fn full() -> Self {
        Self { words: [u64::MAX; WORDS] }
    }

    /// Only the bits of `[start, end)` of `source`.
    pub fn slice(source: &PageMask, start: usize, end: usize) -> Self {
        let mut m = PageMask::empty();
        for w in 0..WORDS {
            m.words[w] = source.words[w] & range_word(w, start, end);
        }
        m
    }

    pub fn any(&self) -> bool {
        self.words.iter().any(|&w| w != 0)
    }

    pub fn none(&self) -> bool {
        !self.any()
    }

    pub fn any_in_range(&self, start: usize, end: usize) -> bool {
        if start >= end || end > PAGES_PER_BLOCK {
            return false;
        }
        (start / 64..=(end - 1) / 64).any(|w| self.words[w] & range_word(w, start, end) != 0)
    }

    pub fn all_in_range(&self, start: usize, end: usize) -> bool {
        if start >= end || end > PAGES_PER_BLOCK {
            return false;
        }
        (start / 64..=(end - 1) / 64).all(|w| {
            let m = range_word(w, start, end);
            self.words[w] & m == m
        })
    }

    pub fn get(&self, i: usize) -> bool {
        self.words[i / 64] & (1 << (i % 64)) != 0
    }

    pub fn set(&mut self, i: usize) {
        self.words[i / 64] |= 1 << (i % 64);
    }

    pub fn unset(&mut self, i: usize) {
        self.words[i / 64] &= !(1 << (i % 64));
    }

    pub fn set_range(&mut self, start: usize, end: usize) {
        for w in 0..WORDS {
            self.words[w] |= range_word(w, start, end);
        }
    }

    pub fn unset_range(&mut self, start: usize, end: usize) {
        for w in 0..WORDS {
            self.words[w] &= !range_word(w, start, end);
        }
    }

    pub fn and(&self, o: &PageMask) -> PageMask {
        let mut m = *self;
        for w in 0..WORDS {
            m.words[w] &= o.words[w];
        }
        m
    }

    pub fn or(&self, o: &PageMask) -> PageMask {
        let mut m = *self;
        for w in 0..WORDS {
            m.words[w] |= o.words[w];
        }
        m
    }

    pub fn xor(&self, o: &PageMask) -> PageMask {
        let mut m = *self;
        for w in 0..WORDS {
            m.words[w] ^= o.words[w];
        }
        m
    }

    pub fn not(&self) -> PageMask {
        let mut m = *self;
        for w in 0..WORDS {
            m.words[w] = !m.words[w];
        }
        m
    }

    pub fn count(&self) -> u32 {
        self.words.iter().map(|w| w.count_ones()).sum()
    }

    fn next_set(&self, start: usize) -> usize {
        let mut i = start;
        while i < PAGES_PER_BLOCK {
            let v = self.words[i / 64] >> (i % 64);
            if v != 0 {
                return i + v.trailing_zeros() as usize;
            }
            i += 64 - i % 64;
        }
        PAGES_PER_BLOCK
    }

    /// First run of set bits at or after `start`: `(start, end)`, or
    /// `(PAGES_PER_BLOCK, PAGES_PER_BLOCK)` when none.
    pub fn first_run_from(&self, start: usize) -> (usize, usize) {
        let first = self.next_set(start);
        if first == PAGES_PER_BLOCK {
            return (first, first);
        }
        (first, self.next_unset(first))
    }

    /// The first clear bit at or after `start` (`PAGES_PER_BLOCK` when
    /// none), a word at a time.
    fn next_unset(&self, start: usize) -> usize {
        let mut i = start;
        while i < PAGES_PER_BLOCK {
            let v = !self.words[i / 64] >> (i % 64);
            if v != 0 {
                return (i + v.trailing_zeros() as usize).min(PAGES_PER_BLOCK);
            }
            i += 64 - i % 64;
        }
        PAGES_PER_BLOCK
    }

    /// Maximal runs of set bits.
    pub fn runs(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        let mut next = 0;
        std::iter::from_fn(move || {
            let r = self.first_run_from(next);
            if r.0 == PAGES_PER_BLOCK {
                return None;
            }
            next = r.1;
            Some(r)
        })
    }

    /// `(first set bit, one past last set bit)`, or None when empty.
    pub fn bounds(&self) -> Option<(usize, usize)> {
        let first = self.next_set(0);
        if first == PAGES_PER_BLOCK {
            return None;
        }
        let last = (0..WORDS).rev().find(|&w| self.words[w] != 0).map(|w| w * 64 + 63 - self.words[w].leading_zeros() as usize)?;
        Some((first, last + 1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_runs_and_bounds() {
        let mut m = PageMask::empty();
        assert!(m.none());
        m.set_range(3, 70);
        m.set(100);
        m.set_range(1000, 1024);
        assert_eq!(m.runs().collect::<Vec<_>>(), vec![(3, 70), (100, 101), (1000, 1024)]);
        assert_eq!(m.bounds(), Some((3, 1024)));
        assert!(m.any_in_range(60, 64));
        assert!(!m.any_in_range(70, 100));
        assert!(m.all_in_range(3, 70));
        assert!(!m.all_in_range(2, 70));
        let s = PageMask::slice(&m, 64, 101);
        assert_eq!(s.runs().collect::<Vec<_>>(), vec![(64, 70), (100, 101)]);
        m.unset_range(0, 1024);
        assert!(m.none());
        assert_eq!(PageMask::full().count(), 1024);
    }

    /// Runs found a word at a time equal the runs of a bit-by-bit walk, on
    /// masks with runs ending at and across word boundaries and at the end.
    #[test]
    fn runs_match_a_bit_walk() {
        let mut rng = 0x2545_F491_4F6C_DD1Du64;
        for round in 0..2000 {
            let mut m = PageMask::empty();
            for _ in 0..(round % 9) {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let s = (rng % PAGES_PER_BLOCK as u64) as usize;
                let len = [1, 2, 63, 64, 65, 128, 300][(rng >> 20) as usize % 7];
                m.set_range(s, (s + len).min(PAGES_PER_BLOCK));
            }
            if round % 5 == 0 {
                m.set_range(PAGES_PER_BLOCK - 64, PAGES_PER_BLOCK);
            }
            let mut walk = Vec::new();
            let mut i = 0;
            while i < PAGES_PER_BLOCK {
                if m.get(i) {
                    let s = i;
                    while i < PAGES_PER_BLOCK && m.get(i) {
                        i += 1;
                    }
                    walk.push((s, i));
                } else {
                    i += 1;
                }
            }
            assert_eq!(m.runs().collect::<Vec<_>>(), walk, "round {round}");
        }
        assert_eq!(PageMask::full().runs().collect::<Vec<_>>(), vec![(0, PAGES_PER_BLOCK)]);
    }
}
