// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Compile waits with the back-end unlocked.
//!
//! The command thread records a draw or dispatch holding the back-end
//! lock. A dispatch (and a draw in the exact mode) whose module or
//! pipeline is still compiling used to wait for it inside that lock, for
//! seconds with a cold shader cache. Every other user of the back-end
//! waited as long: a guest CPU write fault on a GPU-written page (it
//! downloads the page through the back-end), image read-backs, unmaps,
//! flips.
//!
//! With [`record`], the provider waits under the lock only for
//! [`GRACE`]; then it gives the item up unrecorded (a pending compile,
//! like a full staging ring, makes the provider record the item again
//! from the start), the caller releases the lock, waits for compile
//! progress with nothing locked and tries again. `RPS5E_UNLOCKED_COMPILE_WAIT=0`
//! keeps the old in-lock waits.

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Duration;

/// How long one try of an item may wait for compiles under the lock (a
/// pipeline the driver cache holds is ready well within it).
pub const GRACE: Duration = Duration::from_millis(2);

/// Upper bound of one wait outside the lock: the item is tried again at
/// least this often (a finished job also ends the wait at once).
const WAIT_SLICE: Duration = Duration::from_millis(10);

/// Tries of draws and dispatches given up for a pending compile.
pub static PENDING_TRIES: AtomicU64 = AtomicU64::new(0);

thread_local! {
    /// The caller of the provider can wait for compiles unlocked.
    static ALLOWED: Cell<bool> = const { Cell::new(false) };
}

/// `RPS5E_UNLOCKED_COMPILE_WAIT=0`: compile waits stay under the lock.
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_UNLOCKED_COMPILE_WAIT").is_ok_and(|v| v == "0" || v.eq_ignore_ascii_case("off")))
}

/// Whether the provider, called on this thread now, may give an item up
/// for a pending compile.
pub(crate) fn allowed() -> bool {
    ALLOWED.with(Cell::get)
}

/// Run `f` with pending compiles not allowed to give the item up (work
/// recorded in parts, such as the draws of a CPU-read indirect record:
/// a later part cannot be tried again without the earlier ones).
pub(crate) fn forbidden<R>(f: impl FnOnce() -> R) -> R {
    let prev = ALLOWED.with(|a| a.replace(false));
    let r = f();
    ALLOWED.with(|a| a.set(prev));
    r
}

/// Record one item: `attempt` takes the back-end lock, tries the item and
/// releases the lock, returning its result and whether the item was given
/// up for a pending compile. Between tries this waits for compile
/// progress holding nothing. `allow`: the item may be given up at all
/// (false for captures and verification modes that log every try).
pub fn record<R>(allow: bool, mut attempt: impl FnMut() -> (R, bool)) -> R {
    let allow = allow && enabled();
    loop {
        // Read before the try: a job finishing during it ends the wait.
        let seen = super::pool::foreground_done();
        let prev = ALLOWED.with(|a| a.replace(allow));
        let (r, pending) = attempt();
        ALLOWED.with(|a| a.set(prev));
        if !pending || !allow {
            return r;
        }
        PENDING_TRIES.fetch_add(1, Ordering::Relaxed);
        super::pool::wait_foreground_progress(seen, WAIT_SLICE);
    }
}
