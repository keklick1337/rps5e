// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Emulator configurations: the global settings and per-game
//! configurations that replace them for one game (`games.toml` in the
//! configuration directory). Every setting maps onto a `rps5e-cli run`
//! option / environment variable of the emulator.
//!
//! Game configurations can be exported and imported as JSON keyed by title
//! id, every value a string: `{"PPSA01234": {"present_mode": "fifo", ...}}`.

use crate::settings::write_atomic;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub const GAMES_FILE: &str = "games.toml";

pub const DEFAULT_ELF: &str = "eboot.bin";
pub const DEFAULT_CONSOLE_LANGUAGE: i32 = 1;
pub const MAX_CONSOLE_LANGUAGE: i32 = 29;

pub const PRESENT_MODES: [(&str, &str); 3] =
    [("fifo", "FIFO (vsync)"), ("mailbox", "Mailbox (low latency vsync)"), ("immediate", "Immediate (no vsync)")];

pub const LOG_LEVELS: [&str; 5] = ["error", "warn", "info", "debug", "trace"];

/// Guest image scaling in the window (`RPS5E_SCALE_MODE`).
pub const SCALE_MODES: [(&str, &str); 3] =
    [("fit", "Fit (keep aspect ratio)"), ("stretch", "Stretch to the window"), ("integer", "Integer multiples")];

/// Resolution of the emulated display (`RPS5E_DISPLAY_RESOLUTION`).
pub const DISPLAY_RESOLUTIONS: [(&str, &str); 3] =
    [("720p", "720p (1280x720)"), ("1080p", "1080p (1920x1080)"), ("2160p", "2160p (3840x2160)")];

/// The default of [`EmulatorSettings::display_resolution`].
pub const DEFAULT_DISPLAY_RESOLUTION: &str = "2160p";

/// Scaling filter (`RPS5E_SCALE_FILTER`).
pub const SCALE_FILTERS: [(&str, &str); 2] = [("linear", "Linear (smooth)"), ("nearest", "Nearest (sharp pixels)")];

/// `""` = the input settings decide.
pub const INPUT_MODES: [(&str, &str); 5] = [
    ("", "From the input settings"),
    ("auto", "Controller + keyboard + mouse"),
    ("keyboard_mouse", "Keyboard + mouse"),
    ("keyboard", "Keyboard only"),
    ("controller", "Controller only"),
];

/// Console language ids (0..=29).
pub const LANGUAGES: &[(i32, &str)] = &[
    (0, "Japanese"),
    (1, "English (US)"),
    (2, "French"),
    (3, "Spanish"),
    (4, "German"),
    (5, "Italian"),
    (6, "Dutch"),
    (7, "Portuguese (Portugal)"),
    (8, "Russian"),
    (9, "Korean"),
    (10, "Chinese (Traditional)"),
    (11, "Chinese (Simplified)"),
    (12, "Finnish"),
    (13, "Swedish"),
    (14, "Danish"),
    (15, "Norwegian"),
    (16, "Polish"),
    (17, "Portuguese (Brazil)"),
    (18, "English (UK)"),
    (19, "Turkish"),
    (20, "Spanish (Latin America)"),
    (21, "Arabic"),
    (22, "French (Canada)"),
    (23, "Czech"),
    (24, "Hungarian"),
    (25, "Greek"),
    (26, "Romanian"),
    (27, "Thai"),
    (28, "Vietnamese"),
    (29, "Indonesian"),
];

/// Settings of one emulator run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct EmulatorSettings {
    /// fifo | mailbox | immediate (`--present-mode`).
    pub present_mode: String,
    /// Vulkan device index or name substring; empty = automatic (`--gpu`).
    pub gpu_device: String,
    pub fullscreen: bool,
    /// HDR10 output for games that show HDR frames (`--hdr`, `RPS5E_HDR`).
    pub hdr: bool,
    /// 720p | 1080p | 2160p: the emulated display games are told about
    /// (`--display-resolution`, `RPS5E_DISPLAY_RESOLUTION`).
    pub display_resolution: String,
    /// Performance overlay shown from the start (`--perf-overlay`; F10 toggles).
    pub perf_overlay: bool,
    /// Small FPS counter in the corner (`RPS5E_FPS_COUNTER`).
    pub fps_counter: bool,
    /// fit | stretch | integer (`RPS5E_SCALE_MODE`).
    pub scale_mode: String,
    /// linear | nearest (`RPS5E_SCALE_FILTER`).
    pub scale_filter: String,
    /// Per-draw GPU fast paths (`RPS5E_FAST_PATHS=0` turns them off).
    pub fast_paths: bool,
    /// Master volume, percent (`RPS5E_VOLUME`).
    pub volume: u32,
    /// Pause the game while the in-game menu is open (`RPS5E_MENU_PAUSE`).
    pub pause_in_menu: bool,
    /// No window (`--headless`).
    pub headless: bool,
    /// 0..=29 (`--language`).
    pub console_language: i32,
    /// `RUST_LOG` of the emulator.
    pub log_level: String,
    /// No audio device (`--mute`).
    pub mute_audio: bool,
    /// Ignore host input (`--no-input`).
    pub disable_host_input: bool,
    /// Device mode override (`--input-mode`), empty = input settings.
    pub input_mode: String,
    /// Console "skip notice screen" setting (`--skip-notice-screen`).
    pub skip_notice_screen: bool,
    /// Guest display refresh rate in Hz, 0 = default 59.94 (`--refresh-rate`).
    pub refresh_rate: f64,
    /// Apply the game's patch file when present (`--game-patch`).
    pub cheats: bool,
    /// Writable game data (`--data-dir`); empty = default.
    pub data_dir: String,
    /// Save data root (`RPS5E_SAVEDATA_DIR`); empty = `<data dir>/savedata`.
    pub savedata_dir: String,
    /// /temp0 root (`RPS5E_TEMP0_DIR`); empty = per title.
    pub temp0_dir: String,
    /// /download0 root (`RPS5E_DOWNLOAD0_DIR`); empty = per title.
    pub download0_dir: String,
    /// Console system fonts (`RPS5E_SYSTEM_FONTS`); empty = host fonts.
    pub system_fonts_dir: String,
    /// Redirect guest network connections to this address (`RPS5E_NET_REDIRECT`).
    pub net_redirect: String,
    /// Pipeline cache file, `off` disables (`RPS5E_PIPELINE_CACHE`).
    pub pipeline_cache: String,
    /// Further `KEY=VALUE` environment variables.
    pub extra_env: Vec<String>,
    /// Executable inside the game folder.
    pub elf: String,
}

impl Default for EmulatorSettings {
    fn default() -> Self {
        Self {
            present_mode: "mailbox".into(),
            gpu_device: String::new(),
            fullscreen: false,
            hdr: false,
            display_resolution: DEFAULT_DISPLAY_RESOLUTION.into(),
            perf_overlay: false,
            fps_counter: false,
            scale_mode: "fit".into(),
            scale_filter: "linear".into(),
            fast_paths: true,
            volume: 100,
            pause_in_menu: false,
            headless: false,
            console_language: DEFAULT_CONSOLE_LANGUAGE,
            log_level: "info".into(),
            mute_audio: false,
            disable_host_input: false,
            input_mode: String::new(),
            skip_notice_screen: false,
            refresh_rate: 0.0,
            cheats: true,
            data_dir: String::new(),
            savedata_dir: String::new(),
            temp0_dir: String::new(),
            download0_dir: String::new(),
            system_fonts_dir: String::new(),
            net_redirect: String::new(),
            pipeline_cache: String::new(),
            extra_env: Vec::new(),
            elf: DEFAULT_ELF.into(),
        }
    }
}

fn bool_text(b: bool) -> String {
    if b { "true" } else { "false" }.into()
}

fn parse_bool(s: &str) -> Option<bool> {
    match s.trim().to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Some(true),
        "false" | "0" | "no" | "off" => Some(false),
        _ => None,
    }
}

impl EmulatorSettings {
    /// Every setting as text (the import/export form).
    pub fn to_map(&self) -> BTreeMap<String, String> {
        let mut m = BTreeMap::new();
        let mut put = |k: &str, v: String| {
            m.insert(k.to_string(), v);
        };
        put("present_mode", self.present_mode.clone());
        put("gpu_device", self.gpu_device.clone());
        put("fullscreen", bool_text(self.fullscreen));
        put("hdr", bool_text(self.hdr));
        put("display_resolution", self.display_resolution.clone());
        put("perf_overlay", bool_text(self.perf_overlay));
        put("fps_counter", bool_text(self.fps_counter));
        put("scale_mode", self.scale_mode.clone());
        put("scale_filter", self.scale_filter.clone());
        put("fast_paths", bool_text(self.fast_paths));
        put("volume", self.volume.to_string());
        put("pause_in_menu", bool_text(self.pause_in_menu));
        put("headless", bool_text(self.headless));
        put("console_language", self.console_language.to_string());
        put("log_level", self.log_level.clone());
        put("mute_audio", bool_text(self.mute_audio));
        put("disable_host_input", bool_text(self.disable_host_input));
        put("input_mode", self.input_mode.clone());
        put("skip_notice_screen", bool_text(self.skip_notice_screen));
        put("refresh_rate", self.refresh_rate.to_string());
        put("cheats", bool_text(self.cheats));
        put("data_dir", self.data_dir.clone());
        put("savedata_dir", self.savedata_dir.clone());
        put("temp0_dir", self.temp0_dir.clone());
        put("download0_dir", self.download0_dir.clone());
        put("system_fonts_dir", self.system_fonts_dir.clone());
        put("net_redirect", self.net_redirect.clone());
        put("pipeline_cache", self.pipeline_cache.clone());
        put("extra_env", self.extra_env.join(";"));
        put("elf", self.elf.clone());
        m
    }

    /// Set one setting from text.
    pub fn set(&mut self, key: &str, value: &str) -> Result<(), String> {
        let bad = || format!("Invalid value for game setting: {key}");
        let b = |v: &str| parse_bool(v).ok_or_else(bad);
        match key {
            "present_mode" => {
                let v = value.trim().to_ascii_lowercase();
                if !PRESENT_MODES.iter().any(|m| m.0 == v) {
                    return Err(bad());
                }
                self.present_mode = v;
            }
            "gpu_device" => self.gpu_device = value.trim().into(),
            "fullscreen" => self.fullscreen = b(value)?,
            "hdr" => self.hdr = b(value)?,
            "display_resolution" => {
                let v = ps5gpu::display::DisplayResolution::parse(value).ok_or_else(bad)?;
                self.display_resolution = v.as_str().into();
            }
            "perf_overlay" => self.perf_overlay = b(value)?,
            "fps_counter" => self.fps_counter = b(value)?,
            "scale_mode" => {
                let v = value.trim().to_ascii_lowercase();
                if !SCALE_MODES.iter().any(|m| m.0 == v) {
                    return Err(bad());
                }
                self.scale_mode = v;
            }
            "scale_filter" => {
                let v = value.trim().to_ascii_lowercase();
                if !SCALE_FILTERS.iter().any(|m| m.0 == v) {
                    return Err(bad());
                }
                self.scale_filter = v;
            }
            "fast_paths" => self.fast_paths = b(value)?,
            "volume" => {
                let v: u32 = value.trim().parse().map_err(|_| bad())?;
                if v > 100 {
                    return Err(bad());
                }
                self.volume = v;
            }
            "pause_in_menu" => self.pause_in_menu = b(value)?,
            "headless" => self.headless = b(value)?,
            "console_language" => {
                let v: i32 = value.trim().parse().map_err(|_| bad())?;
                if !(0..=MAX_CONSOLE_LANGUAGE).contains(&v) {
                    return Err(bad());
                }
                self.console_language = v;
            }
            "log_level" => {
                let v = value.trim().to_ascii_lowercase();
                if !LOG_LEVELS.contains(&v.as_str()) {
                    return Err(bad());
                }
                self.log_level = v;
            }
            "mute_audio" => self.mute_audio = b(value)?,
            "disable_host_input" => self.disable_host_input = b(value)?,
            "input_mode" => {
                let v = value.trim().to_ascii_lowercase();
                if !INPUT_MODES.iter().any(|m| m.0 == v) {
                    return Err(bad());
                }
                self.input_mode = v;
            }
            "skip_notice_screen" => self.skip_notice_screen = b(value)?,
            "refresh_rate" => {
                let v: f64 = value.trim().parse().map_err(|_| bad())?;
                if v != 0.0 && !(30.0..=360.0).contains(&v) {
                    return Err("Refresh rate must be 0 (default) or between 30 and 360.".into());
                }
                self.refresh_rate = v;
            }
            "cheats" => self.cheats = b(value)?,
            "data_dir" => self.data_dir = value.trim().into(),
            "savedata_dir" => self.savedata_dir = value.trim().into(),
            "temp0_dir" => self.temp0_dir = value.trim().into(),
            "download0_dir" => self.download0_dir = value.trim().into(),
            "system_fonts_dir" => self.system_fonts_dir = value.trim().into(),
            "net_redirect" => self.net_redirect = value.trim().into(),
            "pipeline_cache" => self.pipeline_cache = value.trim().into(),
            "extra_env" => {
                let list: Vec<String> =
                    value.split(';').map(str::trim).filter(|s| !s.is_empty()).map(str::to_string).collect();
                if list.iter().any(|e| !valid_env_entry(e)) {
                    return Err(bad());
                }
                self.extra_env = list;
            }
            "elf" => {
                let v = value.trim();
                if v.is_empty() || v.contains('/') || v.contains('\\') || v == "." || v == ".." {
                    return Err(bad());
                }
                self.elf = v.into();
            }
            _ => return Err(format!("Unknown game setting: {key}")),
        }
        Ok(())
    }

    /// Problems that make the settings unusable.
    pub fn validate(&self) -> Result<(), String> {
        for (k, v) in self.to_map() {
            let mut copy = self.clone();
            copy.set(&k, &v)?;
        }
        Ok(())
    }

    /// Set every key of a JSON object; values must be strings. Unknown keys
    /// and invalid values are errors.
    pub fn apply_json(&mut self, obj: &serde_json::Map<String, serde_json::Value>) -> Result<(), String> {
        let mut next = self.clone();
        for (k, v) in obj {
            let s = v.as_str().ok_or_else(|| format!("Game setting {k} must be a string."))?;
            next.set(k, s)?;
        }
        *self = next;
        Ok(())
    }

    /// The data directory the emulator will use.
    pub fn effective_data_dir(&self) -> PathBuf {
        if self.data_dir.trim().is_empty() {
            ps5hle::filesystem::user_data_dir()
        } else {
            expand_home(self.data_dir.trim())
        }
    }

    /// The save data root the emulator will use.
    pub fn effective_savedata_root(&self) -> PathBuf {
        if self.savedata_dir.trim().is_empty() {
            self.effective_data_dir().join("savedata")
        } else {
            expand_home(self.savedata_dir.trim())
        }
    }

    /// Environment of a run (`input_config`: global input file passed on).
    pub fn env(&self, config_dir: Option<&Path>, input_config: Option<&Path>, patch: Option<&Path>) -> Vec<(String, String)> {
        let mut env = vec![
            ("RUST_LOG".to_string(), self.log_level.clone()),
            ("RPS5E_PRESENT_MODE".to_string(), self.present_mode.clone()),
            ("RPS5E_SYSTEM_LANGUAGE".to_string(), self.console_language.to_string()),
        ];
        let mut flag = |on: bool, k: &str, v: &str| {
            if on {
                env.push((k.to_string(), v.to_string()));
            }
        };
        flag(self.fullscreen, "RPS5E_FULLSCREEN", "1");
        flag(self.hdr, "RPS5E_HDR", "1");
        flag(self.display_resolution != DEFAULT_DISPLAY_RESOLUTION, "RPS5E_DISPLAY_RESOLUTION", &self.display_resolution);
        flag(self.perf_overlay, "RPS5E_PERF_OVERLAY", "1");
        flag(self.fps_counter, "RPS5E_FPS_COUNTER", "1");
        flag(self.scale_mode != "fit", "RPS5E_SCALE_MODE", &self.scale_mode);
        flag(self.scale_filter != "linear", "RPS5E_SCALE_FILTER", &self.scale_filter);
        flag(!self.fast_paths, "RPS5E_FAST_PATHS", "0");
        flag(self.volume != 100, "RPS5E_VOLUME", &self.volume.to_string());
        flag(self.pause_in_menu, "RPS5E_MENU_PAUSE", "1");
        flag(self.headless, "RPS5E_HEADLESS", "1");
        flag(self.mute_audio, "RPS5E_AUDIO", "null");
        flag(self.disable_host_input, "RPS5E_NO_HOST_INPUT", "1");
        flag(self.skip_notice_screen, "RPS5E_SKIP_NOTICE_SCREEN", "1");
        let mut opt = |v: &str, k: &str, path: bool| {
            let v = v.trim();
            if !v.is_empty() {
                let v = if path { expand_home(v).display().to_string() } else { v.to_string() };
                env.push((k.to_string(), v));
            }
        };
        opt(&self.gpu_device, "RPS5E_GPU_DEVICE", false);
        opt(&self.data_dir, "RPS5E_DATA_DIR", true);
        opt(&self.savedata_dir, "RPS5E_SAVEDATA_DIR", true);
        opt(&self.temp0_dir, "RPS5E_TEMP0_DIR", true);
        opt(&self.download0_dir, "RPS5E_DOWNLOAD0_DIR", true);
        opt(&self.system_fonts_dir, "RPS5E_SYSTEM_FONTS", true);
        opt(&self.net_redirect, "RPS5E_NET_REDIRECT", false);
        opt(&self.pipeline_cache, "RPS5E_PIPELINE_CACHE", false);
        if self.refresh_rate != 0.0 {
            env.push(("RPS5E_GUEST_REFRESH_RATE".into(), self.refresh_rate.to_string()));
        }
        if !self.input_mode.is_empty() {
            env.push((ps5input::OVERRIDE_ENV.into(), format!("[device]\nmode = {:?}\n", self.input_mode)));
        }
        if let Some(d) = config_dir {
            env.push(("RPS5E_CONFIG_DIR".into(), d.display().to_string()));
        }
        if let Some(p) = input_config {
            env.push((ps5input::mapping::PATH_ENV.to_string(), p.display().to_string()));
        }
        if let (true, Some(p)) = (self.cheats, patch) {
            env.push(("RPS5E_GAME_PATCH".into(), p.display().to_string()));
        }
        for e in &self.extra_env {
            if let Some((k, v)) = e.split_once('=') {
                env.push((k.trim().to_string(), v.to_string()));
            }
        }
        env
    }
}

fn valid_env_entry(e: &str) -> bool {
    e.split_once('=').is_some_and(|(k, _)| {
        let k = k.trim();
        !k.is_empty() && k.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_')
    })
}

/// `~/x` -> `$HOME/x`.
pub fn expand_home(raw: &str) -> PathBuf {
    match raw.strip_prefix("~/") {
        Some(rest) => home().map(|h| h.join(rest)).unwrap_or_else(|| PathBuf::from(raw)),
        None if raw == "~" => home().unwrap_or_else(|| PathBuf::from(raw)),
        None => PathBuf::from(raw),
    }
}

/// Title ids game configurations are exported and imported under.
pub fn is_game_settings_title_id(id: &str) -> bool {
    id.len() == 9 && id.starts_with("PPSA") && id[4..].bytes().all(|c| c.is_ascii_digit())
}

/// One game's configuration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GameConfig {
    pub title_id: String,
    pub name: String,
    pub settings: EmulatorSettings,
}

/// Per-game configurations by game key (its path).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GameConfigs {
    #[serde(default)]
    pub games: BTreeMap<String, GameConfig>,
}

impl GameConfigs {
    pub fn path(config_dir: &Path) -> PathBuf {
        config_dir.join(GAMES_FILE)
    }

    pub fn parse(text: &str) -> Result<Self, String> {
        toml::from_str(text).map_err(|e| e.to_string())
    }

    pub fn load(config_dir: &Path) -> Result<Self, String> {
        let p = Self::path(config_dir);
        match std::fs::read_to_string(&p) {
            Ok(t) => Self::parse(&t).map_err(|e| format!("{}: {e}", p.display())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(format!("{}: {e}", p.display())),
        }
    }

    pub fn save(&self, config_dir: &Path) -> Result<(), String> {
        write_atomic(&Self::path(config_dir), &toml::to_string_pretty(self).map_err(|e| e.to_string())?)
    }

    /// The settings a game runs with: its own configuration, else the global one.
    pub fn effective<'a>(&'a self, key: &str, global: &'a EmulatorSettings) -> &'a EmulatorSettings {
        self.games.get(key).map(|g| &g.settings).unwrap_or(global)
    }

    pub fn has(&self, key: &str) -> bool {
        self.games.contains_key(key)
    }

    /// Export one game's configuration as `{"<TITLE_ID>": {...}}`.
    pub fn export_json(&self, key: &str) -> Result<String, String> {
        let g = self.games.get(key).ok_or("no configuration for this game")?;
        let id = g.title_id.trim().to_ascii_uppercase();
        if !is_game_settings_title_id(&id) {
            return Err("No game configs with a PPSA code were found.".into());
        }
        let mut root = serde_json::Map::new();
        let obj: serde_json::Map<String, serde_json::Value> =
            g.settings.to_map().into_iter().map(|(k, v)| (k, serde_json::Value::String(v))).collect();
        root.insert(id, serde_json::Value::Object(obj));
        serde_json::to_string_pretty(&serde_json::Value::Object(root)).map_err(|e| e.to_string())
    }
}

/// A parsed import: per title id, the settings to set on top of the base.
#[derive(Debug, PartialEq)]
pub struct Import(pub BTreeMap<String, serde_json::Map<String, serde_json::Value>>);

/// Parse and validate an import file (every entry is checked first).
pub fn parse_import(text: &str) -> Result<Import, String> {
    let doc: serde_json::Value = serde_json::from_str(text)
        .map_err(|_| "Expected a JSON object containing game configs keyed by PPSA code.".to_string())?;
    let root = doc
        .as_object()
        .filter(|o| !o.is_empty())
        .ok_or("Expected a JSON object containing game configs keyed by PPSA code.")?;
    let mut out = BTreeMap::new();
    for (k, v) in root {
        let obj = v.as_object().filter(|o| !o.is_empty());
        let check = obj.ok_or_else(String::new).and_then(|o| EmulatorSettings::default().apply_json(o));
        if !is_game_settings_title_id(k) || check.is_err() {
            return Err(format!("Invalid game config for {k}. {}", check.err().unwrap_or_default()));
        }
        out.insert(k.clone(), obj.cloned().unwrap_or_default());
    }
    Ok(Import(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_map_round_trip() {
        let mut s = EmulatorSettings::default();
        s.present_mode = "fifo".into();
        s.extra_env = vec!["A=1".into(), "B_C=x y".into()];
        s.refresh_rate = 120.0;
        s.scale_mode = "integer".into();
        s.scale_filter = "nearest".into();
        s.volume = 40;
        s.fps_counter = true;
        s.fast_paths = false;
        s.pause_in_menu = true;
        s.hdr = true;
        s.display_resolution = "1080p".into();
        let mut t = EmulatorSettings::default();
        for (k, v) in s.to_map() {
            t.set(&k, &v).unwrap();
        }
        assert_eq!(s, t);
        assert!(s.validate().is_ok());
        assert!(t.set("present_mode", "vsync").is_err());
        assert!(t.set("console_language", "30").is_err());
        assert!(t.set("refresh_rate", "20").is_err());
        assert!(t.set("refresh_rate", "0").is_ok());
        assert!(t.set("fullscreen", "maybe").is_err());
        assert!(t.set("hdr", "maybe").is_err());
        assert!(t.set("hdr", "off").is_ok() && !t.hdr);
        assert!(!EmulatorSettings::default().hdr, "HDR is off by default");
        assert!(t.set("perf_overlay", "on").is_ok() && t.perf_overlay);
        assert_eq!(EmulatorSettings::default().display_resolution, "2160p");
        assert!(t.set("display_resolution", "1440p").is_err());
        assert!(t.set("display_resolution", "").is_err());
        assert!(t.set("display_resolution", " 720P ").is_ok() && t.display_resolution == "720p");
        assert!(t.set("display_resolution", "4k").is_ok() && t.display_resolution == "2160p");
        for (v, _) in DISPLAY_RESOLUTIONS {
            assert!(t.set("display_resolution", v).is_ok() && t.display_resolution == v);
        }
        assert!(t.set("elf", "../x").is_err());
        assert!(t.set("extra_env", "=1").is_err());
        assert!(t.set("nope", "1").unwrap_err().contains("Unknown"));
        assert!(t.set("volume", "101").is_err());
        assert!(t.set("scale_mode", "zoom").is_err());
        assert!(t.set("scale_filter", "NEAREST").is_ok() && t.scale_filter == "nearest");
    }

    #[test]
    fn environment() {
        let mut s = EmulatorSettings::default();
        s.fullscreen = true;
        s.perf_overlay = true;
        s.gpu_device = " 1 ".into();
        s.disable_host_input = true;
        s.input_mode = "keyboard".into();
        s.refresh_rate = 60.0;
        s.extra_env = vec!["RPS5E_GPU_TRACE=1".into()];
        let env = s.env(Some(Path::new("/c")), Some(Path::new("/c/in.toml")), Some(Path::new("/d/p.json")));
        let get = |k: &str| env.iter().find(|(n, _)| n == k).map(|(_, v)| v.as_str());
        assert_eq!(get("RPS5E_FULLSCREEN"), Some("1"));
        assert_eq!(get("RPS5E_PERF_OVERLAY"), Some("1"));
        assert_eq!(get("RPS5E_GPU_DEVICE"), Some("1"));
        assert_eq!(get("RPS5E_NO_HOST_INPUT"), Some("1"));
        assert_eq!(get("RPS5E_PRESENT_MODE"), Some("mailbox"));
        assert_eq!(get("RPS5E_INPUT_CONFIG"), Some("/c/in.toml"));
        assert_eq!(get("RPS5E_CONFIG_DIR"), Some("/c"));
        assert_eq!(get("RPS5E_GAME_PATCH"), Some("/d/p.json"));
        assert_eq!(get("RPS5E_GUEST_REFRESH_RATE"), Some("60"));
        assert_eq!(get("RPS5E_GPU_TRACE"), Some("1"));
        assert!(get("RPS5E_INPUT_OVERRIDE").unwrap().contains("keyboard"));
        assert_eq!(get("RPS5E_HEADLESS"), None);
        // Defaults of the in-game menu settings add nothing.
        for k in ["RPS5E_HDR", "RPS5E_DISPLAY_RESOLUTION", "RPS5E_FPS_COUNTER", "RPS5E_SCALE_MODE", "RPS5E_SCALE_FILTER", "RPS5E_FAST_PATHS", "RPS5E_VOLUME", "RPS5E_MENU_PAUSE"] {
            assert_eq!(get(k), None, "{k}");
        }
        assert_eq!(get("RPS5E_SAVEDATA_DIR"), None);
        // The override must be a valid input file.
        ps5input::mapping::parse_config(get("RPS5E_INPUT_OVERRIDE").unwrap()).unwrap();
        s.hdr = true;
        s.volume = 35;
        s.scale_mode = "stretch".into();
        s.fast_paths = false;
        s.fps_counter = true;
        s.display_resolution = "720p".into();
        let env = s.env(None, None, None);
        let get = |k: &str| env.iter().find(|(n, _)| n == k).map(|(_, v)| v.as_str());
        assert_eq!(get("RPS5E_VOLUME"), Some("35"));
        assert_eq!(get("RPS5E_SCALE_MODE"), Some("stretch"));
        assert_eq!(get("RPS5E_FAST_PATHS"), Some("0"));
        assert_eq!(get("RPS5E_FPS_COUNTER"), Some("1"));
        assert_eq!(get("RPS5E_DISPLAY_RESOLUTION"), Some("720p"));
        assert_eq!(
            ps5gpu::display::DisplayResolution::parse(get("RPS5E_DISPLAY_RESOLUTION").unwrap()),
            Some(ps5gpu::display::DisplayResolution::P720)
        );
        // The emulator reads the value the launcher passes as "on".
        assert!(ps5gpu::display::parse_hdr(get("RPS5E_HDR").unwrap()));
        s.cheats = false;
        let env = s.env(None, None, Some(Path::new("/d/p.json")));
        assert!(!env.iter().any(|(k, _)| k == "RPS5E_GAME_PATCH"));
    }

    #[test]
    fn game_configs_file_and_export_import() {
        let mut c = GameConfigs::default();
        let mut s = EmulatorSettings::default();
        s.fullscreen = true;
        s.display_resolution = "1080p".into();
        c.games.insert("/g/A".into(), GameConfig { title_id: "PPSA01234".into(), name: "A".into(), settings: s.clone() });
        let text = toml::to_string_pretty(&c).unwrap();
        assert_eq!(GameConfigs::parse(&text).unwrap(), c);
        let global = EmulatorSettings::default();
        assert!(c.effective("/g/A", &global).fullscreen);
        assert!(!c.effective("/g/B", &global).fullscreen);
        assert_eq!(c.effective("/g/A", &global).display_resolution, "1080p");
        assert_eq!(c.effective("/g/B", &global).display_resolution, "2160p");
        // A games.toml written before the setting existed reads the default.
        let old = text.replace("display_resolution = \"1080p\"\n", "");
        assert_ne!(old, text);
        assert_eq!(GameConfigs::parse(&old).unwrap().games["/g/A"].settings.display_resolution, "2160p");

        let json = c.export_json("/g/A").unwrap();
        let imp = parse_import(&json).unwrap();
        let mut t = EmulatorSettings::default();
        t.apply_json(&imp.0["PPSA01234"]).unwrap();
        assert_eq!(t, s);

        assert!(parse_import("[]").is_err());
        assert!(parse_import("{}").is_err());
        assert!(parse_import(r#"{"CUSA00001": {"fullscreen": "true"}}"#).is_err());
        assert!(parse_import(r#"{"PPSA00001": {"fullscreen": true}}"#).unwrap_err().contains("must be a string"));
        assert!(parse_import(r#"{"PPSA00001": {"bogus": "1"}}"#).unwrap_err().contains("Unknown"));
        assert!(parse_import(r#"{"PPSA00001": {"fullscreen": "true"}}"#).is_ok());
    }
}

/// The user's home folder (`%USERPROFILE%` on Windows when `HOME` is unset).
fn home() -> Option<PathBuf> {
    std::env::var_os("HOME").or_else(|| if cfg!(windows) { std::env::var_os("USERPROFILE") } else { None }).map(PathBuf::from)
}
