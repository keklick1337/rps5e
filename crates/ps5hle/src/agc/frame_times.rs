// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! `RPS5E_FRAME_TIMES=<file>`: the time of every flip (when the GPU work
//! before it is done and the frame is handed to the display) is kept in
//! memory and written as `<ms since start> <ms since previous flip>` per
//! line, so single long frames (freezes) show up exactly. New lines are
//! appended to the file at most [`APPEND_EVERY_USEC`] after their flip (and
//! the rest at shutdown), so a run that ends without its stop hooks (a
//! crash, a forced exit) loses at most the last second of times.

use std::io::Write;
use std::sync::{Mutex, OnceLock};

struct Log {
    path: std::path::PathBuf,
    usec: Vec<u64>,
    /// Flips already in the file (None until the file was created).
    written: Option<usize>,
    /// Time of the last append.
    last_append: u64,
}

fn log() -> Option<&'static Mutex<Log>> {
    static LOG: OnceLock<Option<Mutex<Log>>> = OnceLock::new();
    LOG.get_or_init(|| {
        let path = std::env::var_os("RPS5E_FRAME_TIMES").filter(|p| !p.is_empty())?;
        ps5vm::shutdown::on_shutdown(ps5vm::shutdown::ORDER_RECORDS, "frame-times", write);
        // An hour at 60 fps without growing.
        Some(Mutex::new(Log { path: path.into(), usec: Vec::with_capacity(216_000), written: None, last_append: 0 }))
    })
    .as_ref()
}

/// Longest time a flip's line waits before it is appended to the file.
const APPEND_EVERY_USEC: u64 = 1_000_000;

/// Record a flip now.
pub fn flip() {
    ps5gpu::backend::ab::note_flip();
    if let Some(l) = log() {
        let t = crate::time::process_time_usec();
        let mut g = l.lock().unwrap_or_else(|e| e.into_inner());
        g.usec.push(t);
        if t.saturating_sub(g.last_append) >= APPEND_EVERY_USEC {
            g.last_append = t;
            append(&mut g);
        }
    }
}

/// Lines of the log for `usec` flip times (with the header).
pub fn format(usec: &[u64]) -> String {
    format_from(usec, 0)
}

/// Lines for the flips from index `from` on (the header when `from` is 0).
fn format_from(usec: &[u64], from: usize) -> String {
    let mut out = String::with_capacity((usec.len() - from.min(usec.len())) * 20 + 48);
    if from == 0 {
        out.push_str("# ms_since_start ms_since_previous_flip\n");
    }
    for i in from..usec.len() {
        let t = usec[i];
        let d = if i == 0 { 0.0 } else { (t - usec[i - 1]) as f64 / 1000.0 };
        out.push_str(&format!("{:.3} {:.3}\n", t as f64 / 1000.0, d));
    }
    out
}

/// Append the flips not yet in the file (the file is created at the first
/// append, replacing an older one).
fn append(l: &mut Log) {
    let from = l.written.unwrap_or(0);
    if l.written.is_some() && from == l.usec.len() {
        return;
    }
    let text = format_from(&l.usec, from);
    let file = if l.written.is_none() {
        std::fs::File::create(&l.path)
    } else {
        std::fs::OpenOptions::new().append(true).open(&l.path)
    };
    match file.and_then(|mut f| f.write_all(text.as_bytes())) {
        Ok(()) => l.written = Some(l.usec.len()),
        Err(e) => log::warn!("frame times: cannot write {}: {e}", l.path.display()),
    }
}

fn write() {
    let Some(l) = log() else { return };
    let mut l = l.lock().unwrap_or_else(|e| e.into_inner());
    append(&mut l);
}

#[cfg(test)]
mod tests {
    #[test]
    fn lines_hold_time_and_interval() {
        assert_eq!(super::format(&[1000, 17_667, 117_667]), "# ms_since_start ms_since_previous_flip\n1.000 0.000\n17.667 16.667\n117.667 100.000\n");
    }

    #[test]
    fn appends_continue_the_file() {
        let dir = std::env::temp_dir().join(format!("rps5e-ft-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("ft.txt");
        let mut l = super::Log { path: path.clone(), usec: vec![1000, 17_667], written: None, last_append: 0 };
        super::append(&mut l);
        l.usec.push(117_667);
        super::append(&mut l);
        super::append(&mut l);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), super::format(&[1000, 17_667, 117_667]));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
