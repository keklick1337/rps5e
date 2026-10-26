// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Where the sequencer cuts its work into slices (deterministic).
//!
//! Hard cuts follow dispatches, end-of-pipe work and flips (serial work in
//! later stages); a submission's end and a drain cut too. Soft cuts keep
//! slices near `N` draws (`RPS5E_SLICE_DRAWS`): once a slice holds `N`
//! draws it ends at the next render-target change, or after `N/4` more
//! draws without one.

use super::reglog::Cut;
use crate::render::regs::cx;

#[derive(Clone, Debug)]
pub struct Cutter {
    budget: u32,
    draws: u32,
    target_changed: bool,
    last_target: u64,
}

impl Cutter {
    pub fn new(budget: u32) -> Cutter {
        Cutter { budget: budget.max(1), draws: 0, target_changed: false, last_target: 0 }
    }

    pub fn budget(&self) -> u32 {
        self.budget
    }

    /// The slice ended (any cut).
    pub fn reset(&mut self) {
        self.draws = 0;
        self.target_changed = false;
    }

    /// The render targets changed (a target base register was written).
    #[inline]
    pub fn target_written(&mut self) {
        self.target_changed = true;
    }

    /// A draw whose targets hash to `key` (callers that see resolved draws
    /// instead of register writes).
    pub fn draw_targets(&mut self, key: u64) {
        if key != self.last_target {
            self.last_target = key;
            self.target_changed = true;
        }
    }

    /// Before a draw: the cut the slice takes in front of it, if any.
    pub fn before_draw(&mut self) -> Option<Cut> {
        let cut = if self.draws < self.budget {
            None
        } else if self.target_changed {
            Some(Cut::Soft)
        } else if self.draws >= self.budget + self.budget / 4 {
            Some(Cut::Forced)
        } else {
            None
        };
        if cut.is_some() {
            self.reset();
        }
        self.target_changed = false;
        self.draws += 1;
        cut
    }
}

/// True when context registers `[offset, offset+n)` hold a render-target
/// base (CB_COLOR0..7_BASE, DB_Z_READ_BASE).
#[inline]
pub fn writes_target_base(offset: u32, n: u32) -> bool {
    const LAST: u32 = cx::CB_COLOR0_BASE + 7 * cx::CB_COLOR_STRIDE;
    if offset > LAST || offset + n <= cx::DB_Z_READ_BASE {
        return false;
    }
    if cx::DB_Z_READ_BASE.wrapping_sub(offset) < n {
        return true;
    }
    (0..8).any(|i| (cx::CB_COLOR0_BASE + i * cx::CB_COLOR_STRIDE).wrapping_sub(offset) < n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn soft_cuts_wait_for_a_target_change_within_a_quarter() {
        let mut c = Cutter::new(8);
        let cuts: Vec<Option<Cut>> = (0..8).map(|_| c.before_draw()).collect();
        assert!(cuts.iter().all(|c| c.is_none()));
        // Over budget without a change: two more draws in the grace.
        assert_eq!(c.before_draw(), None);
        assert_eq!(c.before_draw(), None);
        assert_eq!(c.before_draw(), Some(Cut::Forced));
        // A change after the budget cuts at the next draw.
        for _ in 0..8 {
            assert_eq!(c.before_draw(), None);
        }
        c.target_written();
        assert_eq!(c.before_draw(), Some(Cut::Soft));
        // A change before the budget does not carry over.
        c.target_written();
        assert_eq!(c.before_draw(), None);
        for _ in 0..7 {
            c.before_draw();
        }
        assert_eq!(c.before_draw(), None);
        c.draw_targets(5);
        assert_eq!(c.before_draw(), Some(Cut::Soft));
    }

    #[test]
    fn target_base_registers() {
        assert!(writes_target_base(cx::CB_COLOR0_BASE, 1));
        assert!(writes_target_base(cx::CB_COLOR0_BASE + 3 * cx::CB_COLOR_STRIDE - 2, 4));
        assert!(writes_target_base(cx::DB_Z_READ_BASE, 1));
        assert!(writes_target_base(0, 0x400));
        assert!(!writes_target_base(cx::CB_COLOR0_BASE + 1, 4));
        assert!(!writes_target_base(0x200, 0x10));
        assert!(!writes_target_base(0x390, 8));
    }
}
