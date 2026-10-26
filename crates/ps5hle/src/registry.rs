// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! HLE export registry: maps NID strings to native callbacks.

#[cfg(windows)]
#[allow(unused_imports)]
use crate::host::c as libc;
use crate::context::CpuContext;
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

/// Result of an HLE call. The value is written to RAX.
pub type HleResult = i64;

/// Callback signature: receives a mutable context snapshot, returns RAX.
pub type HleCallback = fn(&mut CpuContext) -> HleResult;

/// One registered export.
#[derive(Clone, Copy)]
pub struct HleExport {
    pub nid: &'static str,
    pub name: &'static str,
    pub library: &'static str,
    pub callback: HleCallback,
    /// Used only when no loaded module exports the symbol natively.
    pub fallback: bool,
    /// The generic catalog stub (returns 0x80020002): no implementation.
    /// A flag, not a comparison with the stub's address: small functions
    /// may be instantiated in other crates, where their address differs.
    pub stub: bool,
}

/// Names for NIDs that only have a stub callback. Stored separately so the
/// generic stub can log the function name on first call.
static STUB_NAMES: Mutex<Option<HashMap<String, String>>> = Mutex::new(None);
static LOGGED_STUBS: Mutex<Option<HashSet<String>>> = Mutex::new(None);

fn generic_stub(ctx: &mut CpuContext) -> HleResult {
    // Unimplemented import: report "not found" rather than fake success.
    // dispatch() logs the first call of each stubbed NID by name.
    let _ = ctx;
    0x8002_0002u32 as i32 as i64
}

/// Central registry mapping NIDs to callbacks.
pub struct HleRegistry {
    by_nid: HashMap<String, HleExport>,
}

impl Default for HleRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl HleRegistry {
    pub fn new() -> Self {
        Self {
            by_nid: HashMap::new(),
        }
    }

    /// Build a registry pre-loaded with all built-in HLE implementations
    /// plus stubs for every other known export.
    pub fn with_defaults() -> Self {
        let mut r = Self::new();
        crate::libc::register(&mut r);
        crate::kernel::register(&mut r);
        crate::memory::register(&mut r);
        crate::time::register(&mut r);
        crate::modules::register(&mut r);
        crate::tls::register(&mut r);
        crate::pthread::register(&mut r);
        crate::signals::register(&mut r);
        crate::filesystem::register(&mut r);
        crate::videoout::register(&mut r);
        crate::cxx::register(&mut r);
        crate::math::register(&mut r);
        crate::system_service::register(&mut r);
        crate::pad::register(&mut r);
        crate::rtc::register(&mut r);
        // System service family.
        crate::user_service::register(&mut r);
        crate::app_content::register(&mut r);
        crate::sysmodule::register(&mut r);
        crate::common_dialog::register(&mut r);
        crate::ime::register(&mut r);
        crate::playgo::register(&mut r);
        crate::playgo_dialog::register(&mut r);
        crate::jpeg_dec::register(&mut r);
        crate::jpeg_enc::register(&mut r);
        crate::savedata::register(&mut r);
        crate::savedata_dialog::register(&mut r);
        crate::np_manager::register(&mut r);
        crate::np_trophy2::register(&mut r);
        crate::np_game_intent::register(&mut r);
        crate::np_party::register(&mut r);
        crate::np_utility::register(&mut r);
        crate::np_entitlement::register(&mut r);
        crate::np_auth::register(&mut r);
        crate::np_uds::register(&mut r);
        crate::np_webapi2::register(&mut r);
        crate::np_commerce::register(&mut r);
        crate::np_signaling::register(&mut r);
        // Kernel synchronization objects (equeues, semaphores, event flags).
        crate::sync::register(&mut r);
        crate::stdio::register(&mut r);
        crate::ampr::register(&mut r);
        // Networking (offline), JSON and small system services.
        crate::net::register(&mut r);
        crate::netctl::register(&mut r);
        crate::posix_webkit::register(&mut r);
        crate::http::register(&mut r);
        crate::ssl::register(&mut r);
        crate::json::register(&mut r);
        crate::share::register(&mut r);
        crate::text_to_speech2::register(&mut r);
        crate::misc_services::register(&mut r);
        crate::firmware_answers::register(&mut r);
        crate::move_controller::register(&mut r);
        crate::vr_tracker2::register(&mut r);
        crate::audio::register(&mut r);
        crate::agc::register(&mut r);
        crate::fiber::register(&mut r);
        crate::ult::register(&mut r);
        crate::kernel_ext::register(&mut r);
        crate::libc_atomic::register(&mut r);
        crate::libc_ext::register(&mut r);
        crate::libc_string::register(&mut r);
        crate::libc_thrd::register(&mut r);
        crate::libc_random::register(&mut r);
        crate::disc_map::register(&mut r);
        crate::videodec::register(&mut r);
        crate::fmod::register(&mut r);
        crate::game_update::register(&mut r);
        crate::game_live_streaming::register(&mut r);
        crate::remoteplay::register(&mut r);
        crate::razor_cpu::register(&mut r);
        crate::player_invitation_dialog::register(&mut r);
        crate::g2p_vr_dialog::register(&mut r);
        crate::signin_dialog::register(&mut r);
        crate::login_dialog::register(&mut r);
        crate::content_export::register(&mut r);
        crate::content_search::register(&mut r);
        crate::convert_keycode::register(&mut r);
        crate::avplayer::register(&mut r);
        crate::videodec2::register(&mut r);
        crate::voice_qos::register(&mut r);
        crate::font::register(&mut r);
        crate::png_dec::register(&mut r);
        crate::png_enc::register(&mut r);
        crate::rudp::register(&mut r);
        crate::voice_chat::register(&mut r);
        crate::psml::register(&mut r);
        crate::zlib::register(&mut r);
        crate::job_manager::register(&mut r);
        r.register_known_stubs();
        r
    }

    pub fn register(&mut self, export: HleExport) {
        self.by_nid.insert(export.nid.to_string(), export);
    }

    /// Register a callback by NID string.
    pub fn register_fn(
        &mut self,
        nid: &'static str,
        name: &'static str,
        library: &'static str,
        cb: HleCallback,
    ) {
        self.register(HleExport {
            nid,
            name,
            library,
            callback: cb,
            fallback: false,
            stub: false,
        });
    }

    /// Register a callback that is only used when no loaded module (e.g. the
    /// game's own libc.prx) exports the symbol; the native body is exact.
    pub fn register_fallback_fn(
        &mut self,
        nid: &'static str,
        name: &'static str,
        library: &'static str,
        cb: HleCallback,
    ) {
        self.register(HleExport {
            nid,
            name,
            library,
            callback: cb,
            fallback: true,
            stub: false,
        });
    }

    /// Register stubs (return-0 callbacks) for every known export in the
    /// built-in catalog that doesn't already have a real implementation.
    /// Each stub's NID is recorded so dispatch() can log its name.
    pub fn register_known_stubs(&mut self) {
        let exports = ps5nid::builtin_exports();
        let mut names = STUB_NAMES.lock().unwrap();
        let map = names.get_or_insert_with(HashMap::new);
        let mut added = 0;
        for e in exports {
            if self.by_nid.contains_key(e.nid) {
                continue;
            }
            // Register the generic stub under this NID.
            self.by_nid.insert(
                e.nid.to_string(),
                HleExport {
                    nid: e.nid,
                    name: e.name,
                    library: e.library,
                    callback: generic_stub,
                    fallback: false,
                    stub: true,
                },
            );
            map.insert(e.nid.to_string(), e.name.to_string());
            added += 1;
        }
        log::info!("[HLE] registered {} stub exports (total now {})", added, self.by_nid.len());
    }

    /// True when `nid` has a real HLE implementation (not the generic
    /// return-0 stub registered for catalog-only exports).
    pub fn is_implemented(&self, nid: &str) -> bool {
        self.by_nid
            .get(nid)
            .is_some_and(|e| !e.stub)
    }

    /// Look up a callback by NID.
    pub fn lookup(&self, nid: &str) -> Option<HleExport> {
        self.by_nid.get(nid).copied()
    }

    /// Dispatch a NID call, writing the result to RAX. Returns false if
    /// the NID is completely unknown.
    pub fn dispatch(&self, nid: &str, ctx: &mut CpuContext) -> bool {
        let entry = match self.by_nid.get(nid) {
            Some(e) => *e,
            None => return false,
        };

        // Debug trace: log every HLE call when RUST_LOG=trace is set.
        // NOTE: log::debug!/trace in signal handler context is NOT
        // async-signal-safe and can crash. Only use log at trace level
        // (which is typically disabled in production).
        let trace = log::log_enabled!(log::Level::Trace);
        if trace {
            log::trace!("[HLE] -> nid={}", nid);
        }

        let result = if profile::enabled() {
            let start = std::time::Instant::now();
            let r = (entry.callback)(ctx);
            profile::record(entry.nid, entry.name, start.elapsed());
            r
        } else {
            (entry.callback)(ctx)
        };
        if trace {
            log::trace!("[HLE] <- nid={} rax={:#x}", nid, result);
        }
        ctx.set(crate::context::CpuRegister::Rax, result as u64);

        // If this was a stub callback, log the first call by name.
        if entry.stub {
            let should_log = {
                let mut logged = LOGGED_STUBS.lock().unwrap();
                let set = logged.get_or_insert_with(HashSet::new);
                set.insert(nid.to_string())
            };
            if should_log {
                let name = STUB_NAMES
                    .lock()
                    .unwrap()
                    .as_ref()
                    .and_then(|m| m.get(nid).cloned())
                    .unwrap_or_else(|| nid.to_string());
                log::info!("[HLE stub] first call: {} ({})", name, nid);
            }
        }

        true
    }

    /// Every registered export with whether it has a real implementation
    /// (not the generic stub), in no particular order.
    pub fn exports(&self) -> impl Iterator<Item = (HleExport, bool)> + '_ {
        self.by_nid.values().map(|e| (*e, !e.stub))
    }

    /// Number of registered exports.
    pub fn len(&self) -> usize {
        self.by_nid.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.by_nid.is_empty()
    }
}

/// Per-export call counts and time (`RPS5E_HLE_PROFILE=<seconds>`): each
/// thread accumulates locally and folds into a shared table about once a
/// second; the table is logged, top 40 by time, every report period.
pub mod profile {
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU8, Ordering};
    use std::sync::Mutex;
    use std::time::{Duration, Instant};

    type Table = HashMap<&'static str, (&'static str, u64, u64)>;

    static STATE: AtomicU8 = AtomicU8::new(0);
    static SHARED: Mutex<Option<(Table, Instant, Duration)>> = Mutex::new(None);

    thread_local! {
        static LOCAL: RefCell<(Table, Option<Instant>)> = RefCell::new((HashMap::new(), None));
    }

    #[inline]
    pub fn enabled() -> bool {
        match STATE.load(Ordering::Relaxed) {
            0 => {
                let period = std::env::var("RPS5E_HLE_PROFILE").ok().and_then(|v| v.parse::<u64>().ok());
                if let Some(p) = period.filter(|&p| p > 0) {
                    *SHARED.lock().unwrap() = Some((HashMap::new(), Instant::now(), Duration::from_secs(p)));
                    STATE.store(2, Ordering::Relaxed);
                    true
                } else {
                    STATE.store(1, Ordering::Relaxed);
                    false
                }
            }
            1 => false,
            _ => true,
        }
    }

    pub fn record(nid: &'static str, name: &'static str, spent: Duration) {
        let _ = LOCAL.try_with(|l| {
            let Ok(mut l) = l.try_borrow_mut() else { return };
            let e = l.0.entry(nid).or_insert((name, 0, 0));
            e.1 += 1;
            e.2 += spent.as_nanos() as u64;
            let now = Instant::now();
            let flush_at = *l.1.get_or_insert(now + Duration::from_secs(1));
            if now >= flush_at {
                l.1 = Some(now + Duration::from_secs(1));
                let local = std::mem::take(&mut l.0);
                drop(l);
                flush(local, now);
            }
        });
    }

    fn flush(local: Table, now: Instant) {
        let mut g = SHARED.lock().unwrap();
        let Some((table, last, period)) = g.as_mut() else { return };
        for (nid, (name, n, ns)) in local {
            let e = table.entry(nid).or_insert((name, 0, 0));
            e.1 += n;
            e.2 += ns;
        }
        if now.duration_since(*last) < *period {
            return;
        }
        let secs = now.duration_since(*last).as_secs_f64();
        *last = now;
        let mut rows: Vec<_> = std::mem::take(table).into_iter().collect();
        rows.sort_by(|a, b| b.1 .2.cmp(&a.1 .2));
        let calls: u64 = rows.iter().map(|r| r.1 .1).sum();
        log::warn!("[hle-profile] {:.1}s: {} calls ({:.0}/s)", secs, calls, calls as f64 / secs);
        for (nid, (name, n, ns)) in rows.iter().take(40) {
            log::warn!(
                "[hle-profile] {:>9.0}/s {:>8.1} ms/s {:>7.2} us/call  {} ({})",
                *n as f64 / secs,
                *ns as f64 / 1e6 / secs,
                *ns as f64 / 1e3 / *n as f64,
                name,
                nid
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every export the HLE registers carries the NID of its name. A wrong
    /// NID leaves an import of the game unbound without any error, so this
    /// covers the implemented callbacks and the catalog stubs together.
    #[test]
    fn every_registered_export_has_the_nid_of_its_name() {
        let reg = HleRegistry::with_defaults();
        let mut checked = 0usize;
        let mut wrong = Vec::new();
        // Implemented exports whose name does not hash to their NID: an
        // internal entry (ProgramExitHandler), and exports whose symbol name
        // no candidate in the NID audit hashes to.
        // A new entry here needs the same kind of reason.
        const KNOWN_MISMATCHES: &[&str] = &[
            "ProgramExitHandler (rps5e#ProgramExitHandler)",
            "FmodSystemSetOutput (uPLTdl3psGk)",
            "scePsmlMfsrGetContextBufferRequirement? (IMlj247LdTo)",
            "scePsmlMfsrCreateContext? (vUk2pWMx3KQ)",
            "scePad_n3kSX62fgNo (n3kSX62fgNo)",
            "sceDiscMapUnknownFJgP (fJgP+wqifno)",
            "sceDiscMapUnknownIoKM (ioKMruft1ek)",
        ];
        let (mut implemented, mut stubs_wrong) = (0usize, 0usize);
        for (e, implemented_here) in reg.exports() {
            if e.name.is_empty() || e.name == "?" {
                continue;
            }
            checked += 1;
            if ps5nid::compute_nid(e.name) != e.nid {
                if implemented_here {
                    wrong.push(format!("{} ({})", e.name, e.nid));
                } else {
                    stubs_wrong += 1;
                }
            }
            implemented += implemented_here as usize;
        }
        eprintln!("checked {checked} named exports ({implemented} implemented), {stubs_wrong} catalog stubs with another NID");
        assert!(checked > 30_000, "only {checked} named exports");
        wrong.retain(|w| !KNOWN_MISMATCHES.contains(&w.as_str()));
        assert!(wrong.is_empty(), "{} exports with the NID of another name: {:?}", wrong.len(), &wrong[..wrong.len().min(20)]);
    }
}
