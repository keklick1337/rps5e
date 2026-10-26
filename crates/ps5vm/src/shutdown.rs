// RustPS5Emu (RPS5E)
// Copyright (C) 2025-2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Process shutdown.
//!
//! Every way the emulator ends goes through [`request`] or [`exit`]: the
//! window closed, SIGINT/SIGTERM/SIGHUP, the guest calling `exit`, a lost
//! GPU device. Guest threads are still running native code at that point,
//! so the process never runs the C `atexit` chain: host libraries (the
//! Vulkan driver first of all) tear themselves down there while guest and
//! GPU threads still call into them, and the resulting fault dumps the
//! whole multi-GiB address space. Instead a shutdown thread
//!
//!   1. raises [`requested`]: HLE waits park guest threads on it, and the
//!      emulator's own threads (GPU command thread, window, audio, input)
//!      stop at their next loop iteration;
//!   2. runs the stop hooks registered with [`on_shutdown`], in `order`;
//!   3. flushes the logger and stdio and leaves with `_exit`.
//!
//! A watchdog ends the process after [`DEADLINE`] if a hook hangs.
//! [`install`] routes SIGINT/SIGTERM/SIGHUP to a signal thread (the
//! signals are blocked in every thread, so no handler ever runs on a guest
//! stack); a second signal exits at once.

use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Longest the stop hooks may take before the process exits anyway.
pub const DEADLINE: Duration = Duration::from_millis(1500);

/// Hook order: files of measurements kept in memory (frame times) are
/// written first, since they take no time and a slow hook after them may
/// run into the [`DEADLINE`]; the GPU command processor stops before the
/// window that presents its output; sound and controller output stop last.
pub const ORDER_RECORDS: u8 = 0;
pub const ORDER_GPU: u8 = 1;
pub const ORDER_WINDOW: u8 = 2;
pub const ORDER_AUDIO: u8 = 3;
pub const ORDER_INPUT: u8 = 4;

/// Exit code when the GPU device was lost.
pub const EXIT_DEVICE_LOST: i32 = 3;

static REQUESTED: AtomicBool = AtomicBool::new(false);
static CODE: AtomicI32 = AtomicI32::new(0);

type Hook = Box<dyn FnOnce() + Send>;
static HOOKS: Mutex<Vec<(u8, &'static str, Hook)>> = Mutex::new(Vec::new());

#[cfg(unix)]
const SIGNALS: [libc::c_int; 3] = [libc::SIGINT, libc::SIGTERM, libc::SIGHUP];

/// True once shutdown has started.
#[inline]
pub fn requested() -> bool {
    REQUESTED.load(Ordering::Acquire)
}

/// Register a stop hook. Hooks run once, on the shutdown thread, sorted by
/// `order`; they must not wait longer than a few hundred milliseconds.
pub fn on_shutdown(order: u8, name: &'static str, hook: impl FnOnce() + Send + 'static) {
    let mut hooks = HOOKS.lock().unwrap_or_else(|e| e.into_inner());
    if requested() {
        // Too late to run with the others: the hooks were already taken.
        return;
    }
    hooks.push((order, name, Box::new(hook)));
}

/// Route Ctrl+C, Ctrl+Break and closing the console to the shutdown path
/// (a second event exits at once).
#[cfg(windows)]
pub fn install() {
    use windows_sys::Win32::System::Console::SetConsoleCtrlHandler;
    static SEEN: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    unsafe extern "system" fn handler(event: u32) -> windows_sys::core::BOOL {
        if SEEN.fetch_add(1, Ordering::AcqRel) == 0 {
            log::info!("shutdown: console event {event} received, stopping (again to force)");
            request(130);
            // The console close event ends the process when the handler
            // returns: give the stop hooks their time.
            if event >= 2 {
                std::thread::sleep(DEADLINE + Duration::from_millis(200));
            }
        } else {
            crate::os::exit_now(130);
        }
        1
    }
    // SAFETY: registering a process-wide console handler.
    unsafe { SetConsoleCtrlHandler(Some(handler), 1) };
}

/// Route SIGINT/SIGTERM/SIGHUP to the shutdown path. Call first thing in
/// `main`, before any thread exists: the signals are blocked in the calling
/// thread and every thread created after it inherits the mask.
#[cfg(unix)]
pub fn install() {
    // SAFETY: plain signal-mask and disposition calls.
    unsafe {
        let mut set: libc::sigset_t = std::mem::zeroed();
        libc::sigemptyset(&mut set);
        for s in SIGNALS {
            libc::sigaddset(&mut set, s);
            // An inherited SIG_IGN (a background job of a script) would
            // drop the signal before the signal thread sees it.
            libc::signal(s, libc::SIG_DFL);
        }
        libc::pthread_sigmask(libc::SIG_BLOCK, &set, std::ptr::null_mut());
    }
    let spawned = std::thread::Builder::new().name("rps5e-signals".into()).spawn(signal_thread);
    if let Err(e) = spawned {
        log::warn!("shutdown: cannot start the signal thread ({e}); signals keep their default action");
        // SAFETY: undo the block so the default action applies.
        unsafe {
            let mut set: libc::sigset_t = std::mem::zeroed();
            libc::sigemptyset(&mut set);
            for s in SIGNALS {
                libc::sigaddset(&mut set, s);
            }
            libc::pthread_sigmask(libc::SIG_UNBLOCK, &set, std::ptr::null_mut());
        }
    }
}

#[cfg(unix)]
fn signal_thread() {
    // SAFETY: waiting on the blocked set.
    let mut set: libc::sigset_t = unsafe { std::mem::zeroed() };
    unsafe {
        libc::sigemptyset(&mut set);
        for s in SIGNALS {
            libc::sigaddset(&mut set, s);
        }
    }
    let mut seen = 0;
    loop {
        // SAFETY: valid set; the info is not needed.
        let sig = unsafe { libc::sigwaitinfo(&set, std::ptr::null_mut()) };
        if sig < 0 {
            continue;
        }
        seen += 1;
        if seen == 1 {
            log::info!("shutdown: {} received, stopping (again to force)", signal_name(sig));
            request(128 + sig);
        } else {
            let msg = b"rps5e: second signal, exiting now\n";
            // SAFETY: raw write and _exit: nothing else is waited for.
            unsafe {
                libc::write(2, msg.as_ptr().cast(), msg.len());
                libc::_exit(128 + sig);
            }
        }
    }
}

#[cfg(unix)]
fn signal_name(sig: libc::c_int) -> &'static str {
    match sig {
        libc::SIGINT => "SIGINT",
        libc::SIGTERM => "SIGTERM",
        libc::SIGHUP => "SIGHUP",
        _ => "signal",
    }
}

/// Start shutdown with exit code `code` (the first request's code wins).
/// Returns at once; the process ends on the shutdown thread.
pub fn request(code: i32) {
    if REQUESTED.swap(true, Ordering::AcqRel) {
        return;
    }
    CODE.store(code, Ordering::Release);
    if std::thread::Builder::new().name("rps5e-shutdown".into()).spawn(finish).is_err() {
        finish();
    }
}

/// [`request`] and never return: the calling thread parks until the
/// process ends.
pub fn exit(code: i32) -> ! {
    request(code);
    loop {
        std::thread::park();
    }
}

/// Park the calling thread for good once shutdown has started (guest
/// threads in HLE waits); returns at once otherwise.
pub fn park_if_requested() {
    if requested() {
        loop {
            std::thread::park();
        }
    }
}

fn finish() {
    let code = CODE.load(Ordering::Acquire);
    let started = Instant::now();
    log::info!("shutdown: stopping (exit code {code})");
    let watchdog = std::thread::Builder::new().name("rps5e-shutdown-wd".into()).spawn(move || {
        std::thread::sleep(DEADLINE);
        log::warn!("shutdown: stop hooks still running after {DEADLINE:?}; exiting");
        log::logger().flush();
        // Immediate process exit; stdio may be locked by a parked thread,
        // so it is not flushed here.
        crate::os::exit_now(code)
    });
    if watchdog.is_err() {
        log::warn!("shutdown: no watchdog thread; stop hooks run unguarded");
    }
    let mut hooks = std::mem::take(&mut *HOOKS.lock().unwrap_or_else(|e| e.into_inner()));
    hooks.sort_by_key(|h| h.0);
    for (_, name, hook) in hooks {
        let t = Instant::now();
        hook();
        log::debug!("shutdown: {name} stopped in {:?}", t.elapsed());
    }
    log::info!("shutdown: done in {:?}", started.elapsed());
    log::logger().flush();
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    // Leave without the atexit chain (see the module docs).
    crate::os::exit_now(code)
}
