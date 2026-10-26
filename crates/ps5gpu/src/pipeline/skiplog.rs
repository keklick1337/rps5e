// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! `RPS5E_DRAW_SKIP_LOG=1`: one log line for every presented frame in
//! which draws reached the provider and were not recorded, with the count
//! per reason and the first few (vertex, pixel) programs.
//!
//! A layer missing from exactly one frame shows up here as a frame line
//! with skips while its neighbours have none. Reasons:
//! - `compiling`: a module or pipeline was still compiling on a worker
//!   (`RPS5E_ASYNC_PIPELINES=0` waits instead);
//! - `skip`: nothing to draw (no topology, zero count, no bound target, an
//!   unsupported primitive, indirect records that could not be read);
//! - `shader`: the front end or compiler could not handle a stage (also a
//!   materialisation that failed on unreadable guest memory);
//! - `error`: a host failure (staging ring, buffer allocation, pipeline
//!   creation).

/// Why a draw was not recorded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SkipReason {
    Compiling = 0,
    Skip = 1,
    Shader = 2,
    Error = 3,
}

const REASONS: [&str; 4] = ["compiling", "skip", "shader", "error"];
/// Programs listed per frame line.
const FIRST: usize = 4;

/// Skipped draws of the frame being recorded.
#[derive(Default)]
pub struct SkipLog {
    counts: [u64; 4],
    first: smallvec::SmallVec<[(SkipReason, &'static str, u64, u64); FIRST]>,
    /// Frame lines produced so far.
    pub lines: u64,
}

/// `RPS5E_DRAW_SKIP_LOG=1`.
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("RPS5E_DRAW_SKIP_LOG").is_ok_and(|v| v == "1"))
}

impl SkipLog {
    /// A draw of programs `vs` / `ps` was not recorded.
    pub fn note(&mut self, reason: SkipReason, vs: u64, ps: u64) {
        self.note_why(reason, "", vs, ps);
    }

    /// [`Self::note`] with the detail of a `skip` (which check refused the
    /// draw), printed as `skip[why]`.
    pub fn note_why(&mut self, reason: SkipReason, why: &'static str, vs: u64, ps: u64) {
        self.counts[reason as usize] += 1;
        if self.first.len() < FIRST && !self.first.iter().any(|&(r, w, v, p)| r == reason && w == why && v == vs && p == ps) {
            self.first.push((reason, why, vs, ps));
        }
    }

    /// End of frame `frame`: its line (None without skips); the counts
    /// start over.
    pub fn frame_line(&mut self, frame: u64) -> Option<String> {
        if self.counts.iter().all(|&c| c == 0) {
            return None;
        }
        let mut line = format!("[draw-skips] frame {frame}:");
        for (name, &n) in REASONS.iter().zip(&self.counts) {
            if n != 0 {
                line.push_str(&format!(" {name}={n}"));
            }
        }
        for &(r, why, vs, ps) in &self.first {
            if why.is_empty() {
                line.push_str(&format!(" | {} vs={vs:#x} ps={ps:#x}", REASONS[r as usize]));
            } else {
                line.push_str(&format!(" | {}[{why}] vs={vs:#x} ps={ps:#x}", REASONS[r as usize]));
            }
        }
        self.counts = [0; 4];
        self.first.clear();
        self.lines += 1;
        Some(line)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_without_skips_have_no_line_and_counts_restart() {
        let mut log = SkipLog::default();
        assert_eq!(log.frame_line(1), None);
        log.note(SkipReason::Compiling, 0x10, 0x20);
        log.note(SkipReason::Compiling, 0x10, 0x20);
        log.note(SkipReason::Shader, 0x30, 0);
        assert_eq!(log.frame_line(2).as_deref(), Some("[draw-skips] frame 2: compiling=2 shader=1 | compiling vs=0x10 ps=0x20 | shader vs=0x30 ps=0x0"));
        assert_eq!(log.frame_line(3), None, "the next frame starts from zero");
        assert_eq!(log.lines, 1);
    }

    #[test]
    fn at_most_four_programs_are_listed() {
        let mut log = SkipLog::default();
        for i in 0..10 {
            log.note(SkipReason::Skip, i, 0);
        }
        let line = log.frame_line(7).unwrap();
        assert!(line.starts_with("[draw-skips] frame 7: skip=10"));
        assert_eq!(line.matches(" | skip ").count(), FIRST);
    }

    #[test]
    fn skip_details_are_printed() {
        let mut log = SkipLog::default();
        log.note_why(SkipReason::Skip, "no targets", 0x10, 0);
        assert_eq!(log.frame_line(1).as_deref(), Some("[draw-skips] frame 1: skip=1 | skip[no targets] vs=0x10 ps=0x0"));
    }
}
