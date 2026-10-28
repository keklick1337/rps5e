// RustPS5Emu (RPS5E)
// Copyright (C) 2025-2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The emulator's name on the host: the environment variable prefix and
//! the per-user data, config and cache folders.

use std::path::{Path, PathBuf};

/// Folder name of the emulator under the host's data/config/cache bases.
pub const APP_DIR: &str = "rps5e";

/// Prefix of every emulator environment variable.
pub const ENV_PREFIX: &str = "RPS5E_";

/// The emulator's folder under a data, config or cache base: `base/rps5e`.
pub fn app_dir(base: &Path) -> PathBuf {
    base.join(APP_DIR)
}

/// The host's per-user cache base (the emulator's caches live in
/// [`app_dir`] of it): `%LOCALAPPDATA%` on Windows, else `$XDG_CACHE_HOME`,
/// else `~/.cache`. None when none of them is set.
pub fn cache_base() -> Option<PathBuf> {
    let var = |name: &str| std::env::var_os(name).filter(|v| !v.is_empty()).map(PathBuf::from);
    #[cfg(windows)]
    if let Some(d) = var("LOCALAPPDATA") {
        return Some(d);
    }
    var("XDG_CACHE_HOME").or_else(|| var("HOME").map(|h| h.join(".cache")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn app_dir_is_the_app_folder_under_the_base() {
        assert_eq!(app_dir(Path::new("/home/u/.cache")), Path::new("/home/u/.cache/rps5e"));
    }
}
