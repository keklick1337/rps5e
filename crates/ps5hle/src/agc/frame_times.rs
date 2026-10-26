// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! `RPS5E_FRAME_TIMES=<file>`: the time of every flip (when the GPU work
//! before it is done and the frame is handed to the display) is kept in
//! memory and written at shutdown as `<ms since start> <ms since previous
//! flip>` per line, so single long frames (freezes) show up exactly. The
//! file is also rewritten every [`REWRITE_EVERY`] flips, so a run that ends
//! without its stop hooks (a crash, a forced exit) still leaves its times.

use std::sync::{Mutex, OnceLock};

struct Log {
    path: std::path::PathBuf,
    usec: Vec<u64>,
}

fn log() -> Option<&'static Mutex<Log>> {
    static LOG: OnceLock<Option<Mutex<Log>>> = OnceLock::new();
    LOG.get_or_init(|| {
        let path = std::env::var_os("RPS5E_FRAME_TIMES").filter(|p| !p.is_empty())?;
        ps5vm::shutdown::on_shutdown(ps5vm::shutdown::ORDER_RECORDS, "frame-times", write);
        // An hour at 60 fps without growing.
        Some(Mutex::new(Log { path: path.into(), usec: Vec::with_capacity(216_000) }))
    })
    .as_ref()
}

/// Flips between two rewrites of the file while running.
const REWRITE_EVERY: usize = 1024;

/// Record a flip now.
pub fn flip() {
    ps5gpu::backend::ab::note_flip();
    if let Some(l) = log() {
        let t = crate::time::process_time_usec();
        let mut g = l.lock().unwrap_or_else(|e| e.into_inner());
        g.usec.push(t);
        if g.usec.len() % REWRITE_EVERY == 0 {
            let (path, text) = (g.path.clone(), format(&g.usec));
            drop(g);
            let _ = std::fs::write(path, text);
        }
    }
}

/// Lines of the log for `usec` flip times.
pub fn format(usec: &[u64]) -> String {
    let mut out = String::with_capacity(usec.len() * 20);
    out.push_str("# ms_since_start ms_since_previous_flip\n");
    let mut prev = None;
    for &t in usec {
        let d = prev.map_or(0.0, |p: u64| (t - p) as f64 / 1000.0);
        out.push_str(&format!("{:.3} {:.3}\n", t as f64 / 1000.0, d));
        prev = Some(t);
    }
    out
}

fn write() {
    let Some(l) = log() else { return };
    let l = l.lock().unwrap_or_else(|e| e.into_inner());
    if let Err(e) = std::fs::write(&l.path, format(&l.usec)) {
        log::warn!("frame times: cannot write {}: {e}", l.path.display());
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn lines_hold_time_and_interval() {
        assert_eq!(super::format(&[1000, 17_667, 117_667]), "# ms_since_start ms_since_previous_flip\n1.000 0.000\n17.667 16.667\n117.667 100.000\n");
    }
}
