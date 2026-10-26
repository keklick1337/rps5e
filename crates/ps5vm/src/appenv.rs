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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn app_dir_is_the_app_folder_under_the_base() {
        assert_eq!(app_dir(Path::new("/home/u/.cache")), Path::new("/home/u/.cache/rps5e"));
    }
}
