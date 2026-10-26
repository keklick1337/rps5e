// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Emulation pause (the in-game menu's "pause"): guest threads stop at
//! their next frame boundary — graphics command buffer submission
//! (`sceAgcDriverSubmit*`), `sceAgcSuspendPoint` and the CPU flip
//! (`sceVideoOutSubmitFlip`) — and wait there in HLE until the pause ends
//! (or the process shuts down). Queued GPU work still completes and the
//! window keeps showing the last frame. Host audio is silenced meanwhile.
//! Threads that never reach those calls keep running.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

static PAUSED: AtomicBool = AtomicBool::new(false);

/// Wait slice of a paused thread.
const SLICE: Duration = Duration::from_millis(10);

pub fn set_paused(on: bool) {
    if set_flag(on) {
        log::info!("emulation {}", if on { "paused" } else { "resumed" });
        crate::audio::host::set_muted(on);
    }
}

/// Set the pause flag; whether it changed.
fn set_flag(on: bool) -> bool {
    PAUSED.swap(on, Ordering::AcqRel) != on
}

pub fn paused() -> bool {
    PAUSED.load(Ordering::Acquire)
}

/// Block the calling guest thread while the emulation is paused
/// (delivering guest signals meanwhile).
pub fn wait_if_paused() {
    wait_while_set(&PAUSED);
}

fn wait_while_set(flag: &AtomicBool) {
    while flag.load(Ordering::Acquire) && !ps5vm::shutdown::requested() {
        crate::signals::sleep(SLICE);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn paused_threads_wait_until_resumed() {
        // A flag of its own: the global one would hold up other tests'
        // submissions (and muting would silence their audio).
        static FLAG: AtomicBool = AtomicBool::new(false);
        wait_while_set(&FLAG);
        FLAG.store(true, Ordering::Release);
        let t = std::thread::spawn(|| {
            let start = Instant::now();
            wait_while_set(&FLAG);
            start.elapsed()
        });
        std::thread::sleep(Duration::from_millis(60));
        FLAG.store(false, Ordering::Release);
        assert!(t.join().unwrap() >= Duration::from_millis(50));
        assert!(!paused());
    }
}
