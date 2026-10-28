// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Debugging options of one run ("Run with..."): log filter, GPU profile,
//! performance overlay, display resolution, shader compile wait, HLE call
//! trace and frame dumps. They go on top of the game's configuration as the environment
//! variables the emulator reads, and the launcher keeps the last ones used
//! per game.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Default interval of a frame dump (every Nth presented frame).
pub const DEFAULT_DUMP_EVERY: u32 = 60;

/// Presets of the log filter (`RUST_LOG`); any filter can be typed.
pub const LOG_PRESETS: [(&str, &str); 6] = [
    ("", "From the game's settings"),
    ("warn", "Warnings and errors"),
    ("info", "Info"),
    ("debug", "Debug (verbose)"),
    ("info,ps5gpu=debug", "Info, GPU debug"),
    ("info,ps5hle=debug", "Info, HLE debug"),
];

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct RunOptions {
    /// `RUST_LOG` filter; empty = the log level of the game's settings.
    pub rust_log: String,
    /// GPU profile in the log every 5 s: 0 off, 1 on, 2 with sub-phases
    /// (`RPS5E_GPU_PROFILE`).
    pub gpu_profile: u8,
    /// Performance overlay from the start (`RPS5E_PERF_OVERLAY`).
    pub perf_overlay: bool,
    /// Emulated display resolution (`RPS5E_DISPLAY_RESOLUTION`): empty =
    /// the game's settings, else 720p, 1080p or 2160p.
    pub display_resolution: String,
    /// Per-frame wait for shaders still compiling: empty = the default
    /// (3 ms), `exact` = draws always wait, else milliseconds
    /// (`RPS5E_COMPILE_WAIT_MS`).
    pub compile_wait: String,
    /// Trace every HLE call to a file (`PS5_HLE_TRACE`).
    pub hle_trace: bool,
    /// Trace file; empty = `<data>/debug/<game>/hle.txt`.
    pub hle_trace_file: String,
    /// Start the trace this many seconds after the first call
    /// (`PS5_HLE_TRACE_AFTER`); 0 = from the start.
    pub hle_trace_after: u32,
    /// Write presented frames as images (`RPS5E_DUMP_FRAMES`).
    pub dump_frames: bool,
    /// Frame folder; empty = `<data>/debug/<game>/frames`.
    pub dump_dir: String,
    /// Every Nth frame (`RPS5E_DUMP_EVERY`).
    pub dump_every: u32,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            rust_log: String::new(),
            gpu_profile: 0,
            perf_overlay: false,
            display_resolution: String::new(),
            compile_wait: String::new(),
            hle_trace: false,
            hle_trace_file: String::new(),
            hle_trace_after: 0,
            dump_frames: false,
            dump_dir: String::new(),
            dump_every: DEFAULT_DUMP_EVERY,
        }
    }
}

/// The folder of a game's debugging output: `<data>/debug/<name>`.
pub fn debug_dir(data_dir: &Path, name: &str) -> PathBuf {
    let safe: String = name.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' { c } else { '_' }).take(40).collect();
    data_dir.join("debug").join(if safe.is_empty() { "game".into() } else { safe })
}

impl RunOptions {
    /// Whether the run differs from a plain Play.
    pub fn is_default(&self) -> bool {
        self.rust_log.trim().is_empty()
            && self.gpu_profile == 0
            && !self.perf_overlay
            && self.display_resolution.trim().is_empty()
            && self.compile_wait.trim().is_empty()
            && !self.hle_trace
            && !self.dump_frames
    }

    /// Problems that stop the run.
    pub fn validate(&self) -> Result<(), String> {
        let w = self.compile_wait.trim();
        if !w.is_empty() && !w.eq_ignore_ascii_case("exact") && w.parse::<u32>().is_err() {
            return Err(format!("Compile wait \"{w}\": milliseconds or exact"));
        }
        if self.rust_log.trim().contains(char::is_whitespace) {
            return Err("The log filter has no spaces (e.g. info,ps5gpu=debug)".into());
        }
        let r = self.display_resolution.trim();
        if !r.is_empty() && ps5gpu::display::DisplayResolution::parse(r).is_none() {
            return Err(format!("Display resolution \"{r}\": 720p, 1080p or 2160p"));
        }
        if self.gpu_profile > 2 {
            return Err("GPU profile: 0, 1 or 2".into());
        }
        if self.dump_frames && self.dump_every == 0 {
            return Err("Frame dump: every 1 or more frames".into());
        }
        Ok(())
    }

    /// The HLE trace file of this run.
    pub fn trace_file(&self, debug_dir: &Path) -> PathBuf {
        match self.hle_trace_file.trim() {
            "" => debug_dir.join("hle.txt"),
            p => absolute(&crate::config::expand_home(p)),
        }
    }

    /// The frame dump folder of this run.
    pub fn frames_dir(&self, debug_dir: &Path) -> PathBuf {
        match self.dump_dir.trim() {
            "" => debug_dir.join("frames"),
            p => absolute(&crate::config::expand_home(p)),
        }
    }

    /// Put the options on top of a run's environment `env` (a variable the
    /// game's settings set already is replaced). `debug_dir`: where the
    /// default trace file and frame folder go.
    pub fn apply(&self, env: &mut Vec<(String, String)>, debug_dir: &Path) {
        let mut set = |k: &str, v: String| {
            env.retain(|(n, _)| n != k);
            env.push((k.to_string(), v));
        };
        if !self.rust_log.trim().is_empty() {
            set("RUST_LOG", self.rust_log.trim().to_string());
        }
        if self.gpu_profile > 0 {
            set("RPS5E_GPU_PROFILE", self.gpu_profile.min(2).to_string());
        }
        if self.perf_overlay {
            set("RPS5E_PERF_OVERLAY", "1".into());
        }
        if let Some(r) = ps5gpu::display::DisplayResolution::parse(&self.display_resolution) {
            set("RPS5E_DISPLAY_RESOLUTION", r.as_str().into());
        }
        match self.compile_wait.trim() {
            "" => {}
            w if w.eq_ignore_ascii_case("exact") => set("RPS5E_COMPILE_WAIT_MS", "exact".into()),
            w => set("RPS5E_COMPILE_WAIT_MS", w.to_string()),
        }
        if self.hle_trace {
            set("PS5_HLE_TRACE", self.trace_file(debug_dir).display().to_string());
            if self.hle_trace_after > 0 {
                set("PS5_HLE_TRACE_AFTER", self.hle_trace_after.to_string());
            }
        }
        if self.dump_frames {
            set("RPS5E_DUMP_FRAMES", self.frames_dir(debug_dir).display().to_string());
            set("RPS5E_DUMP_EVERY", self.dump_every.max(1).to_string());
        }
    }

    /// Make the folders the run writes into.
    pub fn prepare(&self, debug_dir: &Path) -> Result<(), String> {
        let mut dirs = Vec::new();
        if self.hle_trace {
            dirs.extend(self.trace_file(debug_dir).parent().map(Path::to_path_buf));
        }
        if self.dump_frames {
            dirs.push(self.frames_dir(debug_dir));
        }
        for d in dirs {
            std::fs::create_dir_all(&d).map_err(|e| format!("{}: {e}", d.display()))?;
        }
        Ok(())
    }

    /// The enabled options in a few words ("GPU profile, HLE trace").
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        if !self.rust_log.trim().is_empty() {
            parts.push(format!("log {}", self.rust_log.trim()));
        }
        match self.gpu_profile {
            0 => {}
            1 => parts.push("GPU profile".into()),
            _ => parts.push("detailed GPU profile".into()),
        }
        if self.perf_overlay {
            parts.push("overlay".into());
        }
        if let Some(r) = ps5gpu::display::DisplayResolution::parse(&self.display_resolution) {
            parts.push(format!("{} display", r.as_str()));
        }
        match self.compile_wait.trim() {
            "" => {}
            w if w.eq_ignore_ascii_case("exact") => parts.push("exact shaders".into()),
            w => parts.push(format!("compile wait {w} ms")),
        }
        if self.hle_trace {
            parts.push("HLE trace".into());
        }
        if self.dump_frames {
            parts.push(format!("frames every {}", self.dump_every.max(1)));
        }
        parts.join(", ")
    }
}

/// `p` against the current folder when relative (the child runs in the
/// game's folder).
fn absolute(p: &Path) -> PathBuf {
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        std::env::current_dir().map(|d| d.join(p)).unwrap_or_else(|_| p.to_path_buf())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get<'a>(env: &'a [(String, String)], k: &str) -> Option<&'a str> {
        let mut found = env.iter().filter(|(n, _)| n == k);
        let v = found.next().map(|(_, v)| v.as_str());
        assert!(found.next().is_none(), "{k} set twice");
        v
    }

    #[test]
    fn defaults_change_nothing() {
        let o = RunOptions::default();
        assert!(o.is_default() && o.validate().is_ok() && o.summary().is_empty());
        let mut env = vec![("RUST_LOG".to_string(), "info".to_string())];
        o.apply(&mut env, Path::new("/d"));
        assert_eq!(env, [("RUST_LOG".to_string(), "info".to_string())]);
    }

    #[test]
    fn options_become_environment() {
        let dbg = Path::new("/data/debug/PPSA01234");
        let o = RunOptions {
            rust_log: " info,ps5gpu=debug ".into(),
            gpu_profile: 2,
            perf_overlay: true,
            display_resolution: "1080p".into(),
            compile_wait: "Exact".into(),
            hle_trace: true,
            hle_trace_after: 30,
            dump_frames: true,
            dump_every: 10,
            ..Default::default()
        };
        assert!(!o.is_default() && o.validate().is_ok());
        let mut env = vec![
            ("RUST_LOG".to_string(), "warn".to_string()),
            ("RPS5E_PRESENT_MODE".into(), "fifo".into()),
            ("RPS5E_DISPLAY_RESOLUTION".into(), "720p".into()),
        ];
        o.apply(&mut env, dbg);
        assert_eq!(get(&env, "RUST_LOG"), Some("info,ps5gpu=debug"), "replaces the game's level");
        assert_eq!(get(&env, "RPS5E_PRESENT_MODE"), Some("fifo"));
        assert_eq!(get(&env, "RPS5E_GPU_PROFILE"), Some("2"));
        assert_eq!(get(&env, "RPS5E_PERF_OVERLAY"), Some("1"));
        assert_eq!(get(&env, "RPS5E_DISPLAY_RESOLUTION"), Some("1080p"), "replaces the game's resolution");
        assert_eq!(get(&env, "RPS5E_COMPILE_WAIT_MS"), Some("exact"));
        assert_eq!(get(&env, "PS5_HLE_TRACE").map(PathBuf::from), Some(dbg.join("hle.txt")));
        assert_eq!(get(&env, "PS5_HLE_TRACE_AFTER"), Some("30"));
        assert_eq!(get(&env, "RPS5E_DUMP_FRAMES").map(PathBuf::from), Some(dbg.join("frames")));
        assert_eq!(get(&env, "RPS5E_DUMP_EVERY"), Some("10"));
        assert_eq!(o.summary(), "log info,ps5gpu=debug, detailed GPU profile, overlay, 1080p display, exact shaders, HLE trace, frames every 10");
        // Explicit places and a millisecond budget.
        let o = RunOptions {
            compile_wait: "12".into(),
            hle_trace: true,
            hle_trace_file: "/t/trace.txt".into(),
            dump_frames: true,
            dump_dir: "/t/frames".into(),
            ..Default::default()
        };
        let mut env = Vec::new();
        o.apply(&mut env, dbg);
        assert_eq!(get(&env, "RPS5E_COMPILE_WAIT_MS"), Some("12"));
        assert_eq!(get(&env, "PS5_HLE_TRACE"), Some("/t/trace.txt"));
        assert_eq!(get(&env, "PS5_HLE_TRACE_AFTER"), None);
        assert_eq!(get(&env, "RPS5E_DUMP_FRAMES"), Some("/t/frames"));
        assert_eq!(get(&env, "RPS5E_DUMP_EVERY"), Some("60"));
    }

    #[test]
    fn invalid_options() {
        let bad = |f: &dyn Fn(&mut RunOptions)| {
            let mut o = RunOptions::default();
            f(&mut o);
            o.validate().is_err()
        };
        assert!(bad(&|o| o.compile_wait = "soon".into()));
        assert!(bad(&|o| o.rust_log = "info debug".into()));
        assert!(bad(&|o| o.gpu_profile = 3));
        assert!(bad(&|o| o.display_resolution = "1440p".into()));
        assert!(bad(&|o| {
            o.dump_frames = true;
            o.dump_every = 0
        }));
    }

    #[test]
    fn prepare_makes_the_output_folders() {
        let dir = crate::test_dir().join(format!("run-options-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let o = RunOptions { hle_trace: true, dump_frames: true, ..Default::default() };
        o.prepare(&dir).unwrap();
        assert!(dir.join("frames").is_dir() && dir.is_dir());
        assert_eq!(debug_dir(Path::new("/d"), "PPSA01234"), PathBuf::from("/d/debug/PPSA01234"));
        assert_eq!(debug_dir(Path::new("/d"), "a b/c"), PathBuf::from("/d/debug/a_b_c"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
