// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Runs a game as a child `rps5e-cli run <eboot>` process (an emulator
//! crash never takes the frontend down) and collects its output, also into
//! a log file per run (`<data dir>/logs/`, the newest [`KEEP_LOGS`] kept).

use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::{Arc, Mutex};

/// Lines kept in the live log.
pub const MAX_LOG_LINES: usize = 20_000;

/// Run logs kept in the log folder.
pub const KEEP_LOGS: usize = 30;

/// The folder of the run logs: `~/.local/share/rps5e/logs`.
pub fn logs_dir() -> PathBuf {
    ps5hle::filesystem::user_data_dir().join("logs")
}

/// A new log file for a run of `name` in `dir` (older logs beyond
/// [`KEEP_LOGS`] are removed); None when the folder cannot be made.
pub fn new_log_file(dir: &Path, name: &str) -> Option<PathBuf> {
    std::fs::create_dir_all(dir).ok()?;
    let mut old: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "log"))
        .collect();
    // Names start with the time: oldest first.
    old.sort_by_key(|p| p.file_name().map(|n| n.to_os_string()));
    while old.len() >= KEEP_LOGS {
        let _ = std::fs::remove_file(old.remove(0));
    }
    let safe = log_name(name);
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0);
    Some(dir.join(format!("{now:015}-{safe}.log")))
}

/// The file-name part of a run log for `name`.
fn log_name(name: &str) -> String {
    name.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' { c } else { '_' }).take(40).collect()
}

/// The newest run log of `name` in `dir` ([`new_log_file`] names).
pub fn latest_log_file(dir: &Path, name: &str) -> Option<PathBuf> {
    let suffix = format!("-{}.log", log_name(name));
    std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.file_name().is_some_and(|n| n.to_string_lossy().ends_with(&suffix)))
        .max_by_key(|p| p.file_name().map(|n| n.to_os_string()))
}

type LogFile = Arc<Mutex<std::fs::File>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LineKind {
    Info,
    Warn,
    Error,
    /// Frontend messages.
    Frontend,
}

#[derive(Clone, Debug)]
pub struct LogLine {
    pub kind: LineKind,
    pub text: String,
}

/// Classify an emulator log line by its level tag.
pub fn classify(line: &str) -> LineKind {
    let head = &line[..line.len().min(48)];
    if head.contains(" ERROR ") || head.contains("panicked") || line.starts_with("Error") {
        LineKind::Error
    } else if head.contains(" WARN ") {
        LineKind::Warn
    } else {
        LineKind::Info
    }
}

#[derive(Default)]
pub struct Log {
    pub lines: VecDeque<LogLine>,
    /// Bumped on every change (for auto-scroll).
    pub generation: u64,
}

impl Log {
    pub fn push(&mut self, kind: LineKind, text: impl Into<String>) {
        if self.lines.len() == MAX_LOG_LINES {
            self.lines.pop_front();
        }
        self.lines.push_back(LogLine { kind, text: text.into() });
        self.generation += 1;
    }

    pub fn clear(&mut self) {
        self.lines.clear();
        self.generation += 1;
    }

    pub fn text(&self) -> String {
        let mut s = String::new();
        for l in &self.lines {
            s.push_str(&l.text);
            s.push('\n');
        }
        s
    }
}

/// Which lines the log view shows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LevelFilter {
    #[default]
    All,
    /// Warnings and errors (and the launcher's own lines).
    Warnings,
    /// Errors (and the launcher's own lines).
    Errors,
}

impl LevelFilter {
    pub const ALL: [LevelFilter; 3] = [LevelFilter::All, LevelFilter::Warnings, LevelFilter::Errors];

    pub fn label(self) -> &'static str {
        match self {
            LevelFilter::All => "All",
            LevelFilter::Warnings => "Warnings",
            LevelFilter::Errors => "Errors",
        }
    }

    fn admits(self, kind: LineKind) -> bool {
        match self {
            LevelFilter::All => true,
            LevelFilter::Warnings => kind != LineKind::Info,
            LevelFilter::Errors => matches!(kind, LineKind::Error | LineKind::Frontend),
        }
    }
}

impl Log {
    /// Indexes of the lines `level` admits that contain `query` (any case).
    pub fn filtered(&self, level: LevelFilter, query: &str) -> Vec<usize> {
        let q = query.trim().to_lowercase();
        self.lines
            .iter()
            .enumerate()
            .filter(|(_, l)| level.admits(l.kind))
            .filter(|(_, l)| q.is_empty() || l.text.to_lowercase().contains(&q))
            .map(|(i, _)| i)
            .collect()
    }

    /// Counts of warnings and errors.
    pub fn problems(&self) -> (usize, usize) {
        let w = self.lines.iter().filter(|l| l.kind == LineKind::Warn).count();
        let e = self.lines.iter().filter(|l| l.kind == LineKind::Error).count();
        (w, e)
    }

    /// A log file's lines, classified like live output.
    pub fn from_text(text: &str) -> Self {
        let mut log = Log::default();
        for line in text.lines() {
            log.push(classify(line), line.to_string());
        }
        log
    }
}

/// Run logs in `dir`, newest first.
pub fn log_files(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|r| r.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|e| e == "log")).collect())
        .unwrap_or_default();
    files.sort_by_key(|p| std::cmp::Reverse(p.file_name().map(|n| n.to_os_string())));
    files
}

/// The name part and start time (ms since the epoch) of a run log name.
pub fn log_file_info(path: &Path) -> (String, u64) {
    let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    match stem.split_once('-') {
        Some((t, name)) => (name.to_string(), t.parse().unwrap_or(0)),
        None => (stem, 0),
    }
}

pub type SharedLog = Arc<Mutex<Log>>;

pub fn log_push(log: &SharedLog, kind: LineKind, text: impl Into<String>) {
    log.lock().unwrap_or_else(|e| e.into_inner()).push(kind, text);
}

/// A running (or finished) emulator process.
pub struct Running {
    pub child: Child,
    pub title: String,
    pub eboot: PathBuf,
    pub exit: Option<ExitStatus>,
    /// Stopped from the launcher (an exit then is no crash).
    pub stopped: bool,
    /// When it started, and how long it ran once it ended.
    pub started: std::time::Instant,
    pub ran_for: Option<std::time::Duration>,
    /// The command as a shell line (to run it again by hand).
    pub command_line: String,
    /// The run's log file.
    pub log_file: Option<PathBuf>,
}

/// Quote `s` for a POSIX shell.
fn sh_quote(s: &str) -> String {
    if !s.is_empty() && s.bytes().all(|c| c.is_ascii_alphanumeric() || b"-_./=:,+@%".contains(&c)) {
        s.to_string()
    } else {
        format!("'{}'", s.replace('\'', r"'\''"))
    }
}

/// Quote `s` for PowerShell.
fn ps_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "''"))
}

/// The command `binary args` with `env`, started in `cwd`, as one line
/// for the host's shell (POSIX sh, or PowerShell on Windows).
pub fn command_line(binary: &Path, args: &[std::ffi::OsString], env: &[(String, String)], cwd: Option<&Path>, windows: bool) -> String {
    let args: Vec<String> = args.iter().map(|a| a.to_string_lossy().into_owned()).collect();
    if windows {
        let mut parts: Vec<String> = Vec::new();
        if let Some(d) = cwd {
            parts.push(format!("Set-Location {}", ps_quote(&d.display().to_string())));
        }
        for (k, v) in env {
            parts.push(format!("$env:{k} = {}", ps_quote(v)));
        }
        let mut run = format!("& {}", ps_quote(&binary.display().to_string()));
        for a in &args {
            run.push(' ');
            run.push_str(&ps_quote(a));
        }
        parts.push(run);
        parts.join("; ")
    } else {
        let mut line = String::new();
        if let Some(d) = cwd {
            line.push_str(&format!("cd {} && ", sh_quote(&d.display().to_string())));
        }
        for (k, v) in env {
            line.push_str(&format!("{k}={} ", sh_quote(v)));
        }
        line.push_str(&sh_quote(&binary.display().to_string()));
        for a in &args {
            line.push(' ');
            line.push_str(&sh_quote(a));
        }
        line
    }
}

/// The folder a run of `subject` starts in.
fn run_dir(subject: &Path) -> Option<&Path> {
    subject.parent().filter(|d| d.is_dir())
}

fn pump<R: Read + Send + 'static>(stream: R, log: SharedLog, file: Option<LogFile>) {
    let _ = std::thread::Builder::new().name("rps5e-gui-log".into()).spawn(move || {
        let mut r = BufReader::new(stream);
        let mut buf = Vec::new();
        loop {
            buf.clear();
            match r.read_until(b'\n', &mut buf) {
                Ok(0) | Err(_) => break,
                Ok(_) => {
                    let line = String::from_utf8_lossy(&buf);
                    let line = line.trim_end_matches(['\n', '\r']);
                    if let Some(f) = &file {
                        use std::io::Write;
                        let _ = writeln!(f.lock().unwrap_or_else(|e| e.into_inner()), "{line}");
                    }
                    log_push(&log, classify(line), line.to_string());
                }
            }
        }
    });
}

/// Start `binary run <eboot>` with `env`; stdout and stderr go to `log`
/// (and to `log_file`).
pub fn launch(
    binary: &Path,
    eboot: &Path,
    title: &str,
    env: &[(String, String)],
    log: &SharedLog,
    log_file: Option<&Path>,
) -> Result<Running, String> {
    spawn(binary, &run_args(eboot), eboot, title, env, log, log_file)
}

/// The arguments of a game run.
pub fn run_args(eboot: &Path) -> [std::ffi::OsString; 2] {
    [std::ffi::OsString::from("run"), eboot.as_os_str().to_owned()]
}

/// The shell line of [`launch`] with the same arguments.
pub fn launch_command_line(binary: &Path, eboot: &Path, env: &[(String, String)]) -> String {
    command_line(binary, &run_args(eboot), env, run_dir(eboot), cfg!(windows))
}

/// A progress line of `precompile` (`  123/456 (26%), 1m 05s left`):
/// done, total and what follows the counts.
pub fn parse_progress(line: &str) -> Option<(usize, usize, String)> {
    let t = line.trim();
    let (counts, rest) = t.split_once(' ').unwrap_or((t, ""));
    let (done, total) = counts.split_once('/')?;
    let (done, total) = (done.parse().ok()?, total.parse().ok()?);
    rest.starts_with('(').then(|| (done, total, format!(" {rest}")))
}

/// Start `binary <args>` (in the folder of `subject`) with `env`; stdout
/// and stderr go to `log`.
pub fn spawn(
    binary: &Path,
    args: &[std::ffi::OsString],
    subject: &Path,
    title: &str,
    env: &[(String, String)],
    log: &SharedLog,
    log_file: Option<&Path>,
) -> Result<Running, String> {
    let eboot = subject;
    let mut cmd = Command::new(binary);
    cmd.args(args).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    if let Some(dir) = run_dir(eboot) {
        cmd.current_dir(dir);
    }
    for (k, v) in env {
        cmd.env(k, v);
    }
    let mut child = cmd.spawn().map_err(|e| format!("cannot start {}: {e}", binary.display()))?;
    log_push(log, LineKind::Frontend, format!("== {} ({}) started, pid {}", title, eboot.display(), child.id()));
    let file = log_file.and_then(|p| match std::fs::File::create(p) {
        Ok(f) => {
            log_push(log, LineKind::Frontend, format!("== log file {}", p.display()));
            Some(Arc::new(Mutex::new(f)))
        }
        Err(e) => {
            log_push(log, LineKind::Warn, format!("== cannot write {}: {e}", p.display()));
            None
        }
    });
    if let Some(out) = child.stdout.take() {
        pump(out, log.clone(), file.clone());
    }
    if let Some(err) = child.stderr.take() {
        pump(err, log.clone(), file);
    }
    Ok(Running {
        child,
        title: title.to_string(),
        eboot: eboot.to_path_buf(),
        exit: None,
        stopped: false,
        started: std::time::Instant::now(),
        ran_for: None,
        command_line: command_line(binary, args, env, run_dir(eboot), cfg!(windows)),
        log_file: log_file.map(Path::to_path_buf),
    })
}

impl Running {
    /// Poll for exit; logs it once.
    pub fn poll(&mut self, log: &SharedLog) -> Option<ExitStatus> {
        if self.exit.is_none() {
            if let Ok(Some(st)) = self.child.try_wait() {
                self.exit = Some(st);
                self.ran_for = Some(self.started.elapsed());
                log_push(log, LineKind::Frontend, format!("== {} exited: {}", self.title, describe_exit(&st)));
            }
        }
        self.exit
    }

    pub fn stop(&mut self, log: &SharedLog) {
        if self.exit.is_none() {
            let _ = self.child.kill();
            self.stopped = true;
            log_push(log, LineKind::Frontend, format!("== stopping {}", self.title));
        }
    }
}

pub fn describe_exit(st: &ExitStatus) -> String {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(sig) = st.signal() {
            return format!("killed by signal {sig}");
        }
    }
    match st.code() {
        Some(0) => "ok".into(),
        Some(c) => format!("exit code {c}"),
        None => "unknown".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precompile_progress_lines() {
        assert_eq!(parse_progress("  123/456 (26%), 1m 05s left"), Some((123, 456, " (26%), 1m 05s left".into())));
        assert_eq!(parse_progress("  0/0 (100%)"), Some((0, 0, " (100%)".into())));
        assert_eq!(parse_progress("PPSA04263: building x on 32 threads"), None);
        assert_eq!(parse_progress("[2026-10-08T01:32:40Z INFO  ps5gpu] 1/2 done"), None);
    }

    #[test]
    fn classifies_levels() {
        assert_eq!(classify("[2026-10-08T01:32:40Z ERROR ps5hle] boom"), LineKind::Error);
        assert_eq!(classify("[2026-10-08T01:32:40Z WARN  ps5hle] hmm"), LineKind::Warn);
        assert_eq!(classify("[2026-10-08T01:32:40Z INFO  ps5hle] ok"), LineKind::Info);
        assert_eq!(classify("thread 'main' panicked at x"), LineKind::Error);
    }

    #[test]
    fn log_filter_and_search() {
        let log = Log::from_text("[t INFO  ps5gpu] pipeline ready\n[t WARN  ps5hle] slow Path\n[t ERROR ps5hle] boom in path\nplain\n");
        assert_eq!(log.problems(), (1, 1));
        assert_eq!(log.filtered(LevelFilter::All, ""), [0, 1, 2, 3]);
        assert_eq!(log.filtered(LevelFilter::Warnings, ""), [1, 2]);
        assert_eq!(log.filtered(LevelFilter::Errors, ""), [2]);
        assert_eq!(log.filtered(LevelFilter::All, "PATH"), [1, 2], "any case");
        assert_eq!(log.filtered(LevelFilter::Errors, "slow"), Vec::<usize>::new());
        let mut log = log;
        log.push(LineKind::Frontend, "== started");
        assert_eq!(log.filtered(LevelFilter::Errors, ""), [2, 4], "launcher lines stay");
    }

    #[test]
    fn log_file_names() {
        let dir = crate::test_dir().join(format!("logs-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let a = new_log_file(&dir, "PPSA00001").unwrap();
        std::fs::write(&a, "x").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(3));
        let b = new_log_file(&dir, "PPSA00002 Game").unwrap();
        std::fs::write(&b, "y").unwrap();
        assert_eq!(log_files(&dir), [b.clone(), a.clone()]);
        let (name, ms) = log_file_info(&b);
        assert_eq!(name, "PPSA00002_Game");
        assert!(ms > 1_600_000_000_000);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn command_lines_quote_for_the_shell() {
        let env = vec![("RUST_LOG".to_string(), "info,ps5gpu=debug".to_string()), ("T".into(), "it's".into())];
        let args = run_args(Path::new("/g/My Game/eboot.bin"));
        let sh = command_line(Path::new("/usr/bin/rps5e-cli"), &args, &env, Some(Path::new("/g/My Game")), false);
        assert_eq!(sh, r#"cd '/g/My Game' && RUST_LOG=info,ps5gpu=debug T='it'\''s' /usr/bin/rps5e-cli run '/g/My Game/eboot.bin'"#);
        let ps = command_line(Path::new(r"C:\e\rps5e-cli.exe"), &args, &env, None, true);
        assert_eq!(ps, r#"$env:RUST_LOG = 'info,ps5gpu=debug'; $env:T = 'it''s'; & 'C:\e\rps5e-cli.exe' 'run' '/g/My Game/eboot.bin'"#);
    }

    #[test]
    fn log_is_bounded() {
        let mut l = Log::default();
        for i in 0..MAX_LOG_LINES + 5 {
            l.push(LineKind::Info, i.to_string());
        }
        assert_eq!(l.lines.len(), MAX_LOG_LINES);
        assert_eq!(l.lines.front().unwrap().text, "5");
    }

    #[cfg(unix)]
    #[test]
    fn child_output_and_exit_are_captured() {
        let log = SharedLog::default();
        // `sh run <file>` runs the script: prints to both streams and fails.
        let dir = std::env::temp_dir().join(format!("rps5e-gui-launch-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let script = dir.join("eboot.bin");
        std::fs::write(&script, "echo out-line\necho err-line >&2\nexit 3\n").unwrap();
        let wrapper = dir.join("fake-cli.sh");
        std::fs::write(&wrapper, "#!/bin/sh\n[ \"$1\" = run ] || exit 9\nexec sh \"$2\"\n").unwrap();
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&wrapper, std::fs::Permissions::from_mode(0o755)).unwrap();
        let log_file = new_log_file(&dir.join("logs"), "PPSA00001 Test").unwrap();
        assert!(log_file.file_name().unwrap().to_string_lossy().ends_with("-PPSA00001_Test.log"));
        let mut r = launch(&wrapper, &script, "Test", &[("X".into(), "1".into())], &log, Some(&log_file)).unwrap();
        let st = r.child.wait().unwrap();
        r.exit = Some(st);
        assert_eq!(st.code(), Some(3));
        // Reader threads may lag behind the exit.
        for _ in 0..100 {
            if log.lock().unwrap().lines.len() >= 3 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        let text = log.lock().unwrap().text();
        assert!(text.contains("out-line") && text.contains("err-line"), "{text}");
        // The run's log file has the child's output too.
        let mut file = String::new();
        for _ in 0..100 {
            file = std::fs::read_to_string(&log_file).unwrap_or_default();
            if file.contains("out-line") && file.contains("err-line") {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(file.contains("out-line") && file.contains("err-line"), "{file}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
