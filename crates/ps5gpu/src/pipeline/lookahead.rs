// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Compile lookahead: while a draw or dispatch waits for a module or
//! pipeline, the provider asks the command stream for the dispatches
//! submitted after it and queues their compiles, so a scene that meets
//! many new compute programs at once compiles them on every worker instead
//! of one after another (a dispatch always waits for its pipeline: later
//! work may read what it writes).
//!
//! The command processor (the PM4 interpreter) installs a [`Source`] for
//! the duration of each executor call ([`with_source`]); the source decodes
//! the packets after the current one on copies of its state and resolves
//! their dispatches without executing anything (no register, memory or
//! label writes, no waits). The provider only queues compiles for what it
//! finds; the dispatches themselves are recorded later by the normal path,
//! so a wrong guess costs a compile and never changes a frame.
//!
//! `RPS5E_COMPILE_LOOKAHEAD=0` turns it off; `RPS5E_COMPILE_LOOKAHEAD=N`
//! sets how many dispatches ahead are resolved (default [`DEFAULT_LIMIT`]).

use crate::render::ResolvedDispatch;
use std::cell::Cell;

/// The submitted command stream after the current executor call.
pub trait Source {
    /// Up to `limit` dispatches after the current packet, in order,
    /// appended to `out`. Must not change any state.
    fn scan(&self, limit: usize, out: &mut Vec<ResolvedDispatch>);
    /// Where the scan would start (the current packet): a provider waiting
    /// on the same call scans once per new result, not per wait.
    fn position(&self) -> u64;
}

/// Dispatches resolved ahead at most.
pub const DEFAULT_LIMIT: usize = 256;

/// `RPS5E_COMPILE_LOOKAHEAD`: 0 off, N the limit (default on).
pub fn limit() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| parse_limit(std::env::var("RPS5E_COMPILE_LOOKAHEAD").ok().as_deref()))
}

fn parse_limit(v: Option<&str>) -> usize {
    match v.map(str::trim) {
        None | Some("") | Some("1") | Some("on") => DEFAULT_LIMIT,
        Some("0") | Some("off") => 0,
        Some(n) => n.parse().unwrap_or(DEFAULT_LIMIT),
    }
}

/// Whether the command processor installs its source.
pub fn enabled() -> bool {
    limit() != 0
}

type SourcePtr = *const (dyn Source + 'static);

thread_local! {
    static SOURCE: Cell<Option<SourcePtr>> = const { Cell::new(None) };
}

/// Run `f` with `src` as this thread's lookahead source.
pub fn with_source<R>(src: &dyn Source, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<SourcePtr>);
    impl Drop for Restore {
        fn drop(&mut self) {
            SOURCE.with(|s| s.set(self.0));
        }
    }
    // SAFETY: only the lifetime is erased; the pointer is cleared (the
    // previous value restored) before `src`'s borrow ends, panics included.
    let ptr: SourcePtr = unsafe { std::mem::transmute::<*const dyn Source, SourcePtr>(src as *const dyn Source) };
    let _restore = Restore(SOURCE.with(|s| s.replace(Some(ptr))));
    f()
}

/// The source's position (None without one on this thread).
pub fn position() -> Option<u64> {
    let ptr = SOURCE.with(|s| s.get())?;
    // SAFETY: set by `with_source` for the duration of the call that is
    // running further up this thread's stack.
    Some(unsafe { &*ptr }.position())
}

/// Up to `limit` dispatches after the current one (empty without a source
/// on this thread).
pub fn upcoming(limit: usize) -> Vec<ResolvedDispatch> {
    let mut out = Vec::new();
    // Taken while it scans: nothing it calls may scan again.
    let Some(ptr) = SOURCE.with(|s| s.take()) else { return out };
    struct Put(SourcePtr);
    impl Drop for Put {
        fn drop(&mut self) {
            SOURCE.with(|s| s.set(Some(self.0)));
        }
    }
    let _put = Put(ptr);
    // SAFETY: see `position`.
    unsafe { &*ptr }.scan(limit, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fixed(Vec<ResolvedDispatch>, u64);

    impl Source for Fixed {
        fn scan(&self, limit: usize, out: &mut Vec<ResolvedDispatch>) {
            // A nested scan finds no source.
            assert!(upcoming(1).is_empty());
            out.extend(self.0.iter().take(limit).cloned());
        }
        fn position(&self) -> u64 {
            self.1
        }
    }

    #[test]
    fn source_is_scoped_to_the_call() {
        assert!(upcoming(8).is_empty());
        assert_eq!(position(), None);
        let regs = crate::render::regfile::RegisterFile::new();
        let d = crate::render::resolve_dispatch(&regs.banks(), &crate::render::DispatchPacket::default());
        let src = Fixed(vec![d.clone(), d], 0x40);
        with_source(&src, || {
            assert_eq!(position(), Some(0x40));
            assert_eq!(upcoming(8).len(), 2);
            assert_eq!(upcoming(1).len(), 1);
            let inner = Fixed(vec![], 0x80);
            with_source(&inner, || assert_eq!(position(), Some(0x80)));
            assert_eq!(position(), Some(0x40));
        });
        assert!(upcoming(8).is_empty());
    }

    #[test]
    fn limit_values() {
        assert_eq!(parse_limit(None), DEFAULT_LIMIT);
        assert_eq!(parse_limit(Some("off")), 0);
        assert_eq!(parse_limit(Some("0")), 0);
        assert_eq!(parse_limit(Some("32")), 32);
        assert_eq!(parse_limit(Some("on")), DEFAULT_LIMIT);
    }
}
