// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Loading and saving input settings from the frontend: the global
//! `rps5e-input.toml` and per-game `profiles/<TITLE_ID>.toml` in the
//! configuration directory (the files the emulator reads, see
//! `ps5input::profile`).

use crate::settings::write_atomic;
use ps5input::mapping::{self, Config};
use ps5input::profile::{profile_path, valid_title_id, PROFILES_DIR};
use std::path::{Path, PathBuf};

/// Which settings are being edited.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Target {
    Global,
    Game { title_id: String, title: String },
}

/// Settings being edited plus where they are saved.
#[derive(Clone, Debug)]
pub struct InputStore {
    pub dir: PathBuf,
    pub target: Target,
    pub config: Config,
    /// Game target: the per-game profile is used.
    pub profile_enabled: bool,
    /// Edited since the last load/save.
    pub dirty: bool,
}

impl InputStore {
    pub fn global_path(dir: &Path) -> PathBuf {
        dir.join(mapping::FILE_NAME)
    }

    fn load_global(dir: &Path) -> Result<Config, String> {
        let p = Self::global_path(dir);
        if p.is_file() {
            mapping::load_file_onto(Config::default(), &p).map_err(|e| e.0)
        } else {
            Ok(Config::default())
        }
    }

    /// Load the settings of `target` from `dir`.
    pub fn load(dir: &Path, target: Target) -> Result<Self, String> {
        let global = Self::load_global(dir)?;
        let (config, profile_enabled) = match &target {
            Target::Global => (global, false),
            Target::Game { title_id, .. } => {
                if !valid_title_id(title_id) {
                    return Err(format!("'{title_id}' is not a title id"));
                }
                let p = profile_path(&dir.join(PROFILES_DIR), title_id);
                if p.is_file() {
                    let cfg = mapping::load_file_onto(global, &p).map_err(|e| e.0)?;
                    let on = cfg.profile_enabled;
                    (cfg, on)
                } else {
                    (global, false)
                }
            }
        };
        Ok(Self { dir: dir.to_path_buf(), target, config, profile_enabled, dirty: false })
    }

    /// The file this target saves to.
    pub fn path(&self) -> PathBuf {
        match &self.target {
            Target::Global => Self::global_path(&self.dir),
            Target::Game { title_id, .. } => profile_path(&self.dir.join(PROFILES_DIR), title_id),
        }
    }

    /// Whether the settings can be edited (a game's settings only with its
    /// profile on).
    pub fn editable(&self) -> bool {
        matches!(self.target, Target::Global) || self.profile_enabled
    }

    /// Turn a game's profile on (starting from the current settings) or
    /// off (kept on disk, marked disabled).
    pub fn set_profile_enabled(&mut self, on: bool) {
        if self.profile_enabled != on {
            self.profile_enabled = on;
            self.dirty = true;
        }
    }

    pub fn save(&mut self) -> Result<(), String> {
        let path = self.path();
        match self.target {
            Target::Global => write_atomic(&path, &self.config.to_toml_string())?,
            Target::Game { .. } => {
                if self.profile_enabled || path.is_file() {
                    let mut cfg = self.config.clone();
                    cfg.profile_enabled = self.profile_enabled;
                    write_atomic(&path, &cfg.to_toml_string())?;
                }
            }
        }
        self.dirty = false;
        Ok(())
    }

    /// The global file if it exists (passed to the emulator).
    pub fn global_file(dir: &Path) -> Option<PathBuf> {
        Some(Self::global_path(dir)).filter(|p| p.is_file())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ps5input::mapping::DeviceMode;

    fn dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("rps5e-gui-store-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn global_and_profile_round_trip_through_the_emulator_resolver() {
        let d = dir("rt");
        let game = Target::Game { title_id: "PPSA10112".into(), title: "SH".into() };
        // Nothing on disk: defaults, profile off.
        let mut g = InputStore::load(&d, Target::Global).unwrap();
        assert_eq!(g.config, Config::default());
        g.config.mode = DeviceMode::Keyboard;
        g.config.mouse_sensitivity = 2.5;
        g.save().unwrap();

        let mut p = InputStore::load(&d, game.clone()).unwrap();
        assert!(!p.profile_enabled && !p.editable());
        assert_eq!(p.config.mode, DeviceMode::Keyboard);
        // Saving a game without a profile writes nothing.
        p.save().unwrap();
        assert!(!p.path().exists());
        p.set_profile_enabled(true);
        p.config.mode = DeviceMode::Controller;
        p.save().unwrap();

        // The emulator sees the profile for that title and the global file elsewhere.
        let global = InputStore::global_file(&d).unwrap();
        let profiles = d.join(PROFILES_DIR);
        let r = ps5input::profile::resolve_with(Some(&global), Some(&profiles), Some("PPSA10112")).unwrap();
        assert_eq!(r.config.mode, DeviceMode::Controller);
        assert_eq!(r.config.mouse_sensitivity, 2.5);
        let r = ps5input::profile::resolve_with(Some(&global), Some(&profiles), Some("PPSA00002")).unwrap();
        assert_eq!(r.config.mode, DeviceMode::Keyboard);

        // Turning the profile off keeps the file but the emulator ignores it.
        let mut p = InputStore::load(&d, game.clone()).unwrap();
        assert!(p.profile_enabled);
        p.set_profile_enabled(false);
        p.save().unwrap();
        assert!(p.path().is_file());
        let r = ps5input::profile::resolve_with(Some(&global), Some(&profiles), Some("PPSA10112")).unwrap();
        assert_eq!(r.config.mode, DeviceMode::Keyboard);
        let p = InputStore::load(&d, game).unwrap();
        assert!(!p.profile_enabled);
        assert!(InputStore::load(&d, Target::Game { title_id: "../x".into(), title: String::new() }).is_err());
        let _ = std::fs::remove_dir_all(&d);
    }
}
