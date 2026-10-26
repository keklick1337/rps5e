// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! "Report issue": a prefilled new-issue page of the project's issue
//! tracker, opened in the browser. Everything goes into the URL (no
//! account or token is needed); the user reviews the text before sending.

/// The issue tracker's new-issue page.
pub const NEW_ISSUE_URL: &str = "https://github.com/keklick1337/rps5e/issues/new";

/// Upper bound of the whole URL: browsers and the tracker cut longer ones.
pub const MAX_URL_LEN: usize = 7000;

/// Longest log line kept (longer ones are cut).
const MAX_LINE: usize = 300;

/// What a report says about the game, the emulator and the machine.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct IssueReport {
    pub game_title: String,
    pub title_id: Option<String>,
    pub app_version: Option<String>,
    pub emulator_version: String,
    pub os: String,
    pub gpu: Option<String>,
    /// How the last run ended, when it failed (`killed by signal 11`).
    pub exit: Option<String>,
    /// The last lines of the game's log, oldest first.
    pub log_tail: Vec<String>,
}

impl IssueReport {
    /// The issue title: the game and its title ID, and the failure if any.
    pub fn title(&self) -> String {
        let id = self.title_id.as_deref().map(|i| format!(" ({i})")).unwrap_or_default();
        match &self.exit {
            Some(e) => format!("{}{id}: crash ({e})", self.game_title),
            None => format!("{}{id}: ", self.game_title),
        }
    }

    /// The issue body (Markdown) with the newest `lines` log lines.
    fn body(&self, lines: usize) -> String {
        let or_unknown = |v: &Option<String>| v.clone().filter(|s| !s.is_empty()).unwrap_or_else(|| "unknown".into());
        let mut b = String::new();
        b.push_str("### What happened\n\n<!-- What did you do, what did you expect, what happened instead? -->\n\n");
        b.push_str("### System\n\n");
        b.push_str(&format!("- Game: {}\n", self.game_title));
        b.push_str(&format!("- Title ID: {}\n", or_unknown(&self.title_id)));
        b.push_str(&format!("- Game version: {}\n", or_unknown(&self.app_version)));
        b.push_str(&format!("- Emulator: RustPS5Emu {}\n", self.emulator_version));
        b.push_str(&format!("- OS: {}\n", self.os));
        b.push_str(&format!("- GPU: {}\n", or_unknown(&self.gpu)));
        if let Some(e) = &self.exit {
            b.push_str(&format!("- Last run: {e}\n"));
        }
        let start = self.log_tail.len().saturating_sub(lines);
        let tail = &self.log_tail[start..];
        if !tail.is_empty() {
            b.push_str(&format!("\n### Last {} log lines\n\n```\n", tail.len()));
            for l in tail {
                let l = l.trim_end();
                match l.char_indices().nth(MAX_LINE) {
                    Some((cut, _)) => {
                        b.push_str(&l[..cut]);
                        b.push_str("...");
                    }
                    None => b.push_str(l),
                }
                b.push('\n');
            }
            b.push_str("```\n");
        }
        b
    }

    /// The new-issue URL, at most [`MAX_URL_LEN`] characters: log lines
    /// are dropped from the oldest until it fits.
    pub fn url(&self) -> String {
        let title = encode(&self.title());
        let mut lines = self.log_tail.len();
        loop {
            let url = format!("{NEW_ISSUE_URL}?title={title}&body={}", encode(&self.body(lines)));
            if url.len() <= MAX_URL_LEN || lines == 0 {
                return url;
            }
            // Drop about as many lines as the excess needs (at least one).
            let excess = url.len() - MAX_URL_LEN;
            lines = lines.saturating_sub((excess / 200).max(1));
        }
    }
}

/// Percent-encoding of a query value: everything but the unreserved
/// characters of RFC 3986, UTF-8 bytes one by one.
pub fn encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 3);
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// The host system, e.g. `Linux 6.12.1 (Gentoo Linux)` or `Windows 10.0`.
pub fn os_description() -> String {
    #[cfg(target_os = "linux")]
    {
        let kernel = std::fs::read_to_string("/proc/sys/kernel/osrelease").map(|s| s.trim().to_string()).unwrap_or_default();
        let distro = std::fs::read_to_string("/etc/os-release")
            .ok()
            .and_then(|t| t.lines().find_map(|l| l.strip_prefix("PRETTY_NAME=").map(|v| v.trim_matches('"').to_string())));
        match distro {
            Some(d) => format!("Linux {kernel} ({d})"),
            None => format!("Linux {kernel}"),
        }
    }
    #[cfg(windows)]
    {
        // The version as the system reports it to the process.
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        let v = std::process::Command::new("cmd")
            .args(["/C", "ver"])
            .creation_flags(CREATE_NO_WINDOW)
            .output()
            .ok()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .filter(|s| !s.is_empty());
        v.unwrap_or_else(|| "Windows".into())
    }
    #[cfg(not(any(target_os = "linux", windows)))]
    {
        std::env::consts::OS.to_string()
    }
}

/// The last `n` lines of a text file.
pub fn file_tail(path: &std::path::Path, n: usize) -> Vec<String> {
    let Ok(bytes) = std::fs::read(path) else { return Vec::new() };
    let text = String::from_utf8_lossy(&bytes);
    let lines: Vec<&str> = text.lines().collect();
    lines[lines.len().saturating_sub(n)..].iter().map(|l| l.to_string()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report(lines: usize) -> IssueReport {
        IssueReport {
            game_title: "SILENT HILL: The Short Message".into(),
            title_id: Some("PPSA10112".into()),
            app_version: Some("01.000".into()),
            emulator_version: "0.1.0-alpha.1".into(),
            os: "Linux 6.12 (Gentoo)".into(),
            gpu: Some("AMD Radeon RX 7900 XTX".into()),
            exit: Some("killed by signal 11".into()),
            log_tail: (0..lines).map(|i| format!("[{i:05}] WARN some log line with spaces & symbols #{i}")).collect(),
        }
    }

    /// The query value back from its encoding.
    fn decode(s: &str) -> String {
        let b = s.as_bytes();
        let mut out = Vec::new();
        let mut i = 0;
        while i < b.len() {
            if b[i] == b'%' {
                out.push(u8::from_str_radix(&s[i + 1..i + 3], 16).unwrap());
                i += 3;
            } else {
                out.push(b[i]);
                i += 1;
            }
        }
        String::from_utf8(out).unwrap()
    }

    fn param<'a>(url: &'a str, name: &str) -> &'a str {
        let q = url.split_once('?').unwrap().1;
        q.split('&').find_map(|kv| kv.strip_prefix(&format!("{name}="))).unwrap()
    }

    #[test]
    fn encodes_reserved_characters_and_utf8() {
        assert_eq!(encode("a b&c=d/é#"), "a%20b%26c%3Dd%2F%C3%A9%23");
        assert_eq!(encode("A-z_0.9~"), "A-z_0.9~");
    }

    #[test]
    fn url_carries_the_title_and_every_field() {
        let url = report(5).url();
        assert!(url.starts_with("https://github.com/keklick1337/rps5e/issues/new?title="));
        assert_eq!(decode(param(&url, "title")), "SILENT HILL: The Short Message (PPSA10112): crash (killed by signal 11)");
        let body = decode(param(&url, "body"));
        for s in ["PPSA10112", "01.000", "RustPS5Emu 0.1.0-alpha.1", "Linux 6.12 (Gentoo)", "AMD Radeon RX 7900 XTX", "killed by signal 11", "#4"] {
            assert!(body.contains(s), "{s} missing from {body}");
        }
        // Only the query separators are left unencoded.
        assert_eq!(url.matches('&').count(), 1);
        assert_eq!(url.matches('?').count(), 1);
    }

    #[test]
    fn long_logs_are_cut_from_the_oldest_line() {
        let url = report(500).url();
        assert!(url.len() <= MAX_URL_LEN, "{}", url.len());
        let body = decode(param(&url, "body"));
        assert!(body.contains("#499"), "the newest line is kept");
        assert!(!body.contains("#0\n"), "the oldest line is dropped");
    }

    #[test]
    fn report_without_failure_or_log() {
        let r = IssueReport { game_title: "Game".into(), emulator_version: "0.1.0".into(), os: "Linux".into(), ..Default::default() };
        assert_eq!(r.title(), "Game: ");
        let body = decode(param(&r.url(), "body"));
        assert!(body.contains("Title ID: unknown") && body.contains("GPU: unknown") && !body.contains("log lines"));
    }

    #[test]
    fn very_long_lines_are_shortened() {
        let mut r = report(1);
        r.log_tail = vec!["x".repeat(5000)];
        let body = decode(param(&r.url(), "body"));
        assert!(body.contains(&format!("{}...", "x".repeat(MAX_LINE))));
        assert!(!body.contains(&"x".repeat(MAX_LINE + 1)));
    }
}
