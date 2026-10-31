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
//! After the current buffer chain the source also reads the command
//! buffers already queued behind it, on its own queue and the others
//! (`RPS5E_LOOKAHEAD_QUEUED=0`: only the current chain), so compute work
//! submitted in many small buffers compiles together instead of one per
//! wait. Work the game has not written yet cannot be seen: a REWIND whose
//! valid bit is clear ends a buffer's scan (GTA V appends its commands
//! while the GPU runs them and writes the next only after the labels of
//! the work before, so its new pipelines are still met one at a time).
//!
//! The provider may record on a thread of its own (`RPS5E_GPU_RECORD_THREAD`),
//! where no source is installed. The command processor therefore also
//! scans on its own thread whenever it meets a compute program no scan has
//! seen yet ([`scan_unseen`], `RPS5E_LOOKAHEAD_EAGER`, on) and hands what it
//! found to the provider ([`take_offered`]), which queues the
//! compiles when it next records: compiles start as soon as the program is
//! met, before the recording reaches the dispatch.
//!
//! `RPS5E_COMPILE_LOOKAHEAD=0` turns it off; `RPS5E_COMPILE_LOOKAHEAD=N`
//! sets how many dispatches ahead are resolved (default [`DEFAULT_LIMIT`],
//! [`QUEUED_LIMIT`] when queued buffers are read).

use crate::render::ResolvedDispatch;
use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};

/// Why the scan of one command stream ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stop {
    /// The stream ended.
    End,
    /// The dispatch or packet limit.
    Limit,
    /// A flip (a scan that does not read past them).
    Flip,
    /// A wait (`RPS5E_LOOKAHEAD_WAITS=0`).
    Wait,
    /// A packet whose effect needs executing (a conditional, a branch, a
    /// register load the command processor produces).
    Branch,
    /// A malformed or unreadable packet.
    Bad,
}

const STOPS: usize = 6;
static SCANS: [AtomicU64; STOPS] = [const { AtomicU64::new(0) }; STOPS];
/// Queued streams scanned, and dispatches found in all.
static QUEUED_SCANNED: AtomicU64 = AtomicU64::new(0);
static FOUND: AtomicU64 = AtomicU64::new(0);

/// Count one stream's scan ending at `stop` (`queued`: a buffer queued
/// after the current one).
pub fn note_scan(stop: Stop, queued: bool) {
    SCANS[stop as usize].fetch_add(1, Ordering::Relaxed);
    if queued {
        QUEUED_SCANNED.fetch_add(1, Ordering::Relaxed);
    }
}

/// `ahead-scans=[end 3 limit 0 flip 0 wait 0 branch 1 bad 0; queued 7;
/// found 120]` for the pipeline line.
pub fn stats_text() -> String {
    let names = ["end", "limit", "flip", "wait", "branch", "bad"];
    let stops: Vec<String> = names.iter().zip(&SCANS).map(|(n, c)| format!("{n} {}", c.load(Ordering::Relaxed))).collect();
    let branches: Vec<String> = BRANCH_OPS.iter().enumerate().filter_map(|(op, c)| Some((op, c.load(Ordering::Relaxed))).filter(|x| x.1 != 0)).map(|(op, n)| format!("{op:#04x}:{n}")).collect();
    format!(
        " ahead-scans=[{}; queued {}; found {}; eager {}; branch ops {}]",
        stops.join(" "),
        QUEUED_SCANNED.load(Ordering::Relaxed),
        FOUND.load(Ordering::Relaxed),
        EAGER_SCANS.load(Ordering::Relaxed),
        branches.join(",")
    )
}

/// `RPS5E_LOOKAHEAD_EAGER=0`: the command processor does not scan ahead
/// on its own when it meets an unseen compute program (only waits scan).
pub fn eager() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| enabled() && !std::env::var("RPS5E_LOOKAHEAD_EAGER").is_ok_and(|v| v == "0" || v.eq_ignore_ascii_case("off")))
}

/// Compute programs some scan has seen (or a dispatch has used): one not
/// in it starts an eager scan.
static SEEN_PROGRAMS: std::sync::Mutex<Option<crate::fxhash::HashSet<u64>>> = std::sync::Mutex::new(None);

/// Dispatches eager scans found, for the provider to queue.
static OFFERED: std::sync::Mutex<Vec<ResolvedDispatch>> = std::sync::Mutex::new(Vec::new());
static HAS_OFFERED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static EAGER_SCANS: AtomicU64 = AtomicU64::new(0);

/// Offered dispatches waiting for the provider at most (older ones are
/// dropped: the provider is far behind, and the scan will meet them again).
const OFFERED_MAX: usize = 8192;

/// For a dispatch of `program` about to run with `src` as the stream
/// after it: when no scan has seen the program, scan now and offer what
/// is found ([`take_offered`]). Cheap when the program was seen.
pub fn scan_unseen(program: u64, src: &dyn Source) {
    if !eager() {
        return;
    }
    {
        let mut g = SEEN_PROGRAMS.lock().unwrap_or_else(|e| e.into_inner());
        let seen = g.get_or_insert_with(Default::default);
        if !seen.insert(program) {
            return;
        }
        if seen.len() > 1 << 16 {
            seen.clear();
        }
    }
    EAGER_SCANS.fetch_add(1, Ordering::Relaxed);
    let mut found = Vec::new();
    src.scan(limit(), &mut found);
    FOUND.fetch_add(found.len() as u64, Ordering::Relaxed);
    if found.is_empty() {
        return;
    }
    {
        let mut g = SEEN_PROGRAMS.lock().unwrap_or_else(|e| e.into_inner());
        let seen = g.get_or_insert_with(Default::default);
        seen.extend(found.iter().map(|d| d.shader.address));
    }
    let mut o = OFFERED.lock().unwrap_or_else(|e| e.into_inner());
    if o.len() + found.len() > OFFERED_MAX {
        let drop = (o.len() + found.len() - OFFERED_MAX).min(o.len());
        o.drain(..drop);
    }
    o.extend(found);
    HAS_OFFERED.store(true, Ordering::Release);
}

/// The dispatches eager scans found since the last call (cheap when none).
pub fn take_offered() -> Vec<ResolvedDispatch> {
    if !HAS_OFFERED.load(Ordering::Acquire) {
        return Vec::new();
    }
    HAS_OFFERED.store(false, Ordering::Release);
    std::mem::take(&mut *OFFERED.lock().unwrap_or_else(|e| e.into_inner()))
}

/// What makes two upcoming dispatches the same compile: program, user
/// data, group shape, resource words and wave size.
pub fn identity(d: &ResolvedDispatch) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = rustc_hash::FxHasher::default();
    (d.shader.address, &d.shader.user_data, d.threads, d.shader.rsrc1, d.shader.rsrc2, d.packet.dispatch_initiator & (1 << 15)).hash(&mut h);
    h.finish()
}

/// Packets that end a stream's scan by default (their effect needs
/// executing: a conditional, a branch, a register the command processor
/// loads) counted by opcode, for the `ahead-scans=` line.
static BRANCH_OPS: [AtomicU64; 256] = [const { AtomicU64::new(0) }; 256];

/// Count a branch-like packet of `opcode` the scan met.
pub fn note_branch(opcode: u32) {
    BRANCH_OPS[(opcode & 0xFF) as usize].fetch_add(1, Ordering::Relaxed);
}

/// `RPS5E_LOOKAHEAD_GUESS=0`: the scan of a stream ends at a packet whose
/// effect needs executing (COND_EXEC, predication, a conditional
/// INDIRECT_BUFFER, COPY_DATA into a register, INDEX_ATTRIBUTES_INDIRECT).
/// By default it reads on past them as if the conditional code ran and the
/// register kept its value (a wrong guess costs a compile); a REWIND whose
/// valid bit is clear still ends it (the packets after it may not be
/// written yet).
pub fn guess_branches() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_LOOKAHEAD_GUESS").is_ok_and(|v| v == "0" || v.eq_ignore_ascii_case("off")))
}

/// `RPS5E_LOOKAHEAD_QUEUED=0`: the scan reads only the current buffer
/// chain (and stops at a flip), not the buffers queued after it.
pub fn queued_streams() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_LOOKAHEAD_QUEUED").is_ok_and(|v| v == "0" || v.eq_ignore_ascii_case("off")))
}

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

/// The default limit when queued buffers are read too.
pub const QUEUED_LIMIT: usize = 1024;

/// `RPS5E_COMPILE_LOOKAHEAD`: 0 off, N the limit (default on).
pub fn limit() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| match parse_limit(std::env::var("RPS5E_COMPILE_LOOKAHEAD").ok().as_deref()) {
        DEFAULT_LIMIT if std::env::var_os("RPS5E_COMPILE_LOOKAHEAD").is_none() && queued_streams() => QUEUED_LIMIT,
        n => n,
    })
}

fn parse_limit(v: Option<&str>) -> usize {
    match v.map(str::trim) {
        None | Some("") | Some("1") | Some("on") => DEFAULT_LIMIT,
        Some("0") | Some("off") => 0,
        Some(n) => n.parse().unwrap_or(DEFAULT_LIMIT),
    }
}

/// `RPS5E_LOOKAHEAD_WAITS=0`: the scan stops at the first packet that
/// waits (memory or semaphore waits); by default it reads on past them,
/// since a wait changes no register the dispatches after it are resolved
/// from.
pub fn past_waits() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_LOOKAHEAD_WAITS").is_ok_and(|v| v == "0" || v.eq_ignore_ascii_case("off")))
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
    FOUND.fetch_add(out.len() as u64, Ordering::Relaxed);
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
