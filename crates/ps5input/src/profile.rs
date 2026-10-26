// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Per-game input profiles: `profiles/<TITLE_ID>.toml` next to the global
//! `rps5e-input.toml` (by default `~/.config/rps5e/profiles/`). A profile
//! has the global file's format and is read on top of the global settings,
//! so it only needs the settings that differ. `[profile] enabled = false`
//! turns a profile off without deleting it; without a profile (or for an
//! unknown title) the global settings apply.

use crate::mapping::{self, Config, ConfigError};
use std::path::{Path, PathBuf};

pub const PROFILES_DIR: &str = "profiles";

/// A title id usable as a profile name: four letters and five digits
/// (`PPSA10112`, `CUSA00001`).
pub fn valid_title_id(id: &str) -> bool {
    let b = id.as_bytes();
    b.len() == 9 && b[..4].iter().all(u8::is_ascii_uppercase) && b[4..].iter().all(u8::is_ascii_digit)
}

/// Profile directory for a global file at `global` (or the default
/// configuration directory when there is none).
pub fn profiles_dir(global: Option<&Path>) -> Option<PathBuf> {
    match global.and_then(Path::parent) {
        Some(p) if !p.as_os_str().is_empty() => Some(p.join(PROFILES_DIR)),
        _ => mapping::config_dir().map(|d| d.join(PROFILES_DIR)),
    }
}

pub fn profile_path(dir: &Path, title_id: &str) -> PathBuf {
    dir.join(format!("{title_id}.toml"))
}

/// Effective settings and the files they came from.
#[derive(Clone, Debug)]
pub struct Resolved {
    pub config: Config,
    pub global: Option<PathBuf>,
    /// The per-game profile in effect, if any.
    pub profile: Option<PathBuf>,
    /// A problem with the profile (the global settings were used).
    pub profile_error: Option<ConfigError>,
}

/// Resolve the settings for `title_id`: the global file (or built-in
/// defaults) overlaid with the title's enabled profile from `profiles`.
/// An invalid global file is an error; an invalid profile falls back to
/// the global settings and is reported in `profile_error`.
pub fn resolve_with(global: Option<&Path>, profiles: Option<&Path>, title_id: Option<&str>) -> Result<Resolved, ConfigError> {
    let base = match global {
        Some(p) => mapping::load_file_onto(Config::default(), p)?,
        None => Config::default(),
    };
    let mut out = Resolved { config: base.clone(), global: global.map(Path::to_path_buf), profile: None, profile_error: None };
    let (Some(dir), Some(id)) = (profiles, title_id.filter(|t| valid_title_id(t))) else {
        return Ok(out);
    };
    let path = profile_path(dir, id);
    if !path.is_file() {
        return Ok(out);
    }
    match mapping::load_file_onto(base, &path) {
        Ok(cfg) if cfg.profile_enabled => {
            out.config = cfg;
            out.profile = Some(path);
        }
        Ok(_) => {}
        Err(e) => out.profile_error = Some(e),
    }
    // A profile's enable switch is not a global setting.
    out.config.profile_enabled = true;
    Ok(out)
}

/// Resolve from the standard locations (see [`mapping::config_candidates`]).
pub fn resolve(title_id: Option<&str>) -> Result<Resolved, ConfigError> {
    let global = mapping::global_config_path();
    let profiles = profiles_dir(global.as_deref());
    resolve_with(global.as_deref(), profiles.as_deref(), title_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mapping::DeviceMode;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("ps5input-profile-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(d.join(PROFILES_DIR)).unwrap();
        d
    }

    #[test]
    fn title_ids() {
        assert!(valid_title_id("PPSA10112"));
        assert!(valid_title_id("CUSA00001"));
        assert!(!valid_title_id("ppsa10112"));
        assert!(!valid_title_id("PPSA1011"));
        assert!(!valid_title_id("../etc/x"));
    }

    #[test]
    fn profile_overlays_global_and_falls_back() {
        let d = tmp("overlay");
        let global = d.join("rps5e-input.toml");
        std::fs::write(&global, "[device]\nmode = \"keyboard\"\n[mouse]\nsensitivity = 3\n").unwrap();
        let profiles = d.join(PROFILES_DIR);
        std::fs::write(profile_path(&profiles, "PPSA10112"), "[device]\nmode = \"controller\"\n").unwrap();

        // Title with a profile: profile settings win, others come from the global file.
        let r = resolve_with(Some(&global), Some(&profiles), Some("PPSA10112")).unwrap();
        assert_eq!(r.config.mode, DeviceMode::Controller);
        assert_eq!(r.config.mouse_sensitivity, 3.0);
        assert_eq!(r.profile.as_deref(), Some(profile_path(&profiles, "PPSA10112").as_path()));

        // Other title / no title: global.
        for t in [Some("PPSA00001"), None] {
            let r = resolve_with(Some(&global), Some(&profiles), t).unwrap();
            assert_eq!(r.config.mode, DeviceMode::Keyboard);
            assert!(r.profile.is_none());
        }

        // Disabled profile: global.
        std::fs::write(profile_path(&profiles, "PPSA10112"), "[device]\nmode = \"controller\"\n[profile]\nenabled = false\n")
            .unwrap();
        let r = resolve_with(Some(&global), Some(&profiles), Some("PPSA10112")).unwrap();
        assert_eq!(r.config.mode, DeviceMode::Keyboard);
        assert!(r.profile.is_none());

        // Broken profile: global, with the error reported.
        std::fs::write(profile_path(&profiles, "PPSA10112"), "[device]\nmode = \"joystick\"\n").unwrap();
        let r = resolve_with(Some(&global), Some(&profiles), Some("PPSA10112")).unwrap();
        assert_eq!(r.config.mode, DeviceMode::Keyboard);
        assert!(r.profile_error.unwrap().0.contains("device.mode"));

        // No global file: built-in defaults under the profile.
        std::fs::write(profile_path(&profiles, "PPSA10112"), "[mouse]\ninvert_y = true\n").unwrap();
        let r = resolve_with(None, Some(&profiles), Some("PPSA10112")).unwrap();
        assert!(r.config.mouse_invert_y);
        assert_eq!(r.config.mode, DeviceMode::Auto);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn profiles_dir_follows_the_global_file() {
        assert_eq!(profiles_dir(Some(Path::new("/x/y/in.toml"))), Some(PathBuf::from("/x/y/profiles")));
    }
}
