// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Shader objects the game creates, and the prewarm work of their programs.
//!
//! A game creates a shader object (`sceAgcCreateShader`) before the draws
//! and dispatches that run it. The object names the program's code, but
//! not the module: a module is a permutation of the program with the
//! resources and state of the draws that use it, so nothing can be
//! translated for it yet. The prewarm list of an earlier run holds those
//! permutations, keyed by program address. While the list builds on the
//! compile workers (in the background during play, or after the
//! translator changed), the modules and pipelines of a program the game
//! has just created move to the front of the list ([`note`]): they are
//! the ones its next draws need, so they are ready before those draws
//! instead of in list order. Urgent work (draws waiting now) still goes
//! first.
//!
//! The provider that queues the list registers each queued job under the
//! program addresses it builds ([`register`]) and installs the pool's
//! [`PromoteHandle`] ([`install`]); a program created before its jobs are
//! queued is promoted when they are. `RPS5E_CREATE_TIME_SHADERS=0` turns
//! it off.

use super::pool::PromoteHandle;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// `RPS5E_CREATE_TIME_SHADERS` (default on): prewarm work of a program
/// moves to the front of the list when the game creates its shader.
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_CREATE_TIME_SHADERS").is_ok_and(|v| v.trim() == "0"))
}

#[derive(Default)]
struct Registry {
    promote: Option<PromoteHandle>,
    /// Queued prewarm jobs by the program addresses they build.
    tags: HashMap<u64, Vec<u64>>,
    /// Programs created so far.
    created: HashSet<u64>,
}

static REGISTRY: Mutex<Option<Registry>> = Mutex::new(None);
static CREATED: AtomicU64 = AtomicU64::new(0);
static PROMOTED: AtomicU64 = AtomicU64::new(0);

/// Jobs to promote, and the handle that does it (called without the lock).
type Promotion = Option<(PromoteHandle, Vec<u64>)>;

impl Registry {
    fn note(&mut self, address: u64) -> Promotion {
        if !self.created.insert(address) {
            return None;
        }
        CREATED.fetch_add(1, Ordering::Relaxed);
        let p = self.promote.clone()?;
        Some((p, self.tags.remove(&address).unwrap_or_default()))
    }

    fn install(&mut self, promote: PromoteHandle) {
        self.tags.clear();
        self.promote = Some(promote);
    }

    fn register(&mut self, tag: u64, addresses: &[u64]) -> Promotion {
        let p = self.promote.clone()?;
        if addresses.iter().any(|a| self.created.contains(a)) {
            return Some((p, vec![tag]));
        }
        for &a in addresses.iter().filter(|&&a| a != 0) {
            self.tags.entry(a).or_default().push(tag);
        }
        None
    }
}

fn with(f: impl FnOnce(&mut Registry) -> Promotion) {
    let p = {
        let mut g = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
        f(g.get_or_insert_with(Registry::default))
    };
    if let Some((promote, tags)) = p {
        for tag in tags {
            if promote.promote(tag) {
                PROMOTED.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

/// The game created a shader object whose program starts at `address`.
pub fn note(address: u64) {
    if enabled() && address != 0 {
        with(|r| r.note(address));
    }
}

/// Use `promote` for the jobs registered from now on (a new provider's
/// pool); jobs registered for an earlier pool are forgotten.
pub fn install(promote: PromoteHandle) {
    if enabled() {
        with(|r| {
            r.install(promote);
            None
        });
    }
}

/// A prewarm job tagged `tag` that builds work of the programs at
/// `addresses` is queued: promoted now when one of them was created.
pub fn register(tag: u64, addresses: &[u64]) {
    if enabled() {
        with(|r| r.register(tag, addresses));
    }
}

/// `(shader objects created, prewarm jobs moved to the front)`.
pub fn counts() -> (u64, u64) {
    (CREATED.load(Ordering::Relaxed), PROMOTED.load(Ordering::Relaxed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn run(p: Promotion, log: &Mutex<Vec<u64>>) {
        if let Some((h, tags)) = p {
            for t in tags {
                if h.promote(t) {
                    log.lock().unwrap().push(t);
                }
            }
        }
    }

    /// Jobs of a program created before they were queued are promoted
    /// when queued; jobs queued first are promoted when the program is
    /// created; other programs' jobs stay where they are; a program
    /// created twice promotes once.
    #[test]
    fn created_programs_move_their_prewarm_jobs_forward() {
        let promoted = Arc::new(Mutex::new(Vec::new()));
        let handle = || PromoteHandle::from_fn(|_| true);
        let mut r = Registry::default();
        // Nothing to promote before a pool is installed.
        assert!(r.note(0x1000).is_none());
        r.install(handle());
        run(r.register(1, &[0x1000, 0]), &promoted);
        run(r.register(2, &[0x2000, 0x3000]), &promoted);
        run(r.register(3, &[0x4000]), &promoted);
        assert_eq!(*promoted.lock().unwrap(), vec![1]);
        run(r.note(0x3000), &promoted);
        run(r.note(0x3000), &promoted);
        assert_eq!(*promoted.lock().unwrap(), vec![1, 2]);
        // A new provider's pool forgets the old pool's jobs.
        r.install(handle());
        promoted.lock().unwrap().clear();
        run(r.note(0x4000), &promoted);
        assert!(promoted.lock().unwrap().is_empty());
        run(r.register(4, &[0x4000]), &promoted);
        assert_eq!(*promoted.lock().unwrap(), vec![4]);
    }
}
