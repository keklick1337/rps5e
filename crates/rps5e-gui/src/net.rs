// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Opt-in downloads (update feed, online cheat collection) through the
//! host's `curl`; the launcher makes no network requests unless the user
//! turns them on.

use std::process::Command;

/// Download `url` (https only), at most `limit` bytes, 30 s timeout.
pub fn fetch(url: &str, limit: u64) -> Result<Vec<u8>, String> {
    if !url.starts_with("https://") {
        return Err("only https URLs are fetched".into());
    }
    let out = Command::new("curl")
        .args(["--silent", "--show-error", "--fail", "--location", "--proto", "=https", "--proto-redir", "=https"])
        .args(["--max-time", "30", "--max-filesize", &limit.to_string(), "--user-agent", "rps5e-launcher"])
        .args(["-H", "Accept: application/vnd.github+json"])
        .arg(url)
        .output()
        .map_err(|e| format!("cannot run curl: {e}"))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr).trim().to_string();
        return Err(if err.is_empty() { format!("download failed ({})", out.status) } else { err });
    }
    if out.stdout.len() as u64 > limit {
        return Err("download exceeds the size limit".into());
    }
    Ok(out.stdout)
}

/// Open a folder or URL with the desktop's handler.
pub fn open_external(target: &str) -> Result<(), String> {
    let opener = if cfg!(target_os = "macos") {
        "open"
    } else if cfg!(windows) {
        "explorer"
    } else {
        "xdg-open"
    };
    Command::new(opener).arg(target).spawn().map(|_| ()).map_err(|e| format!("cannot run {opener}: {e}"))
}

#[cfg(test)]
mod tests {
    #[test]
    fn refuses_plain_http() {
        assert!(super::fetch("http://example.com", 10).is_err());
        assert!(super::fetch("file:///etc/passwd", 10).is_err());
    }
}
