// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Launcher settings, persisted to `~/.config/rps5e/gui.toml`. Input
//! settings live in the emulator's own `rps5e-input.toml` (and
//! `profiles/<TITLE_ID>.toml`), per-game emulator configurations in
//! `games.toml`, all in the same directory.

use crate::config::EmulatorSettings;
use crate::library::Game;
use crate::run_options::RunOptions;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub const FILE_NAME: &str = "gui.toml";

/// Overrides the default library folder.
pub const GAMES_DIR_ENV: &str = "GAMES_DIR";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Theme {
    /// Follow the desktop.
    #[default]
    System,
    Dark,
    Light,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LibraryView {
    #[default]
    Grid,
    List,
}

/// A library folder: searched for games while enabled; a disabled one
/// stays in the list (and in the settings) without being scanned.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LibraryFolder {
    pub path: PathBuf,
    #[serde(default = "enabled_by_default")]
    pub enabled: bool,
}

fn enabled_by_default() -> bool {
    true
}

impl LibraryFolder {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into(), enabled: true }
    }
}

/// How the last run of a game ended (shown in the game's details).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct LastRun {
    /// Seconds since the Unix epoch when the run ended.
    pub ended: u64,
    /// How long it ran, seconds.
    pub seconds: u64,
    /// "ok", "exit code 3", "killed by signal 11", "stopped".
    pub result: String,
    /// Ended without an error (or was stopped from the launcher).
    pub ok: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GuiSettings {
    /// Scan the folder named by `$GAMES_DIR` (when it is set).
    pub default_library: bool,
    /// Library folders scanned for games and packages; each can be turned
    /// off without removing it.
    pub library_folders: Vec<LibraryFolder>,
    /// Single games added by hand: game folders, package files or
    /// executables.
    pub extra_games: Vec<PathBuf>,
    /// Games removed from the library (not shown until restored).
    pub hidden_games: Vec<HiddenGame>,
    /// Emulator binary; empty = this program (or `rps5e-cli` next to it).
    pub emulator_path: String,
    pub theme: Theme,
    pub view: LibraryView,
    /// Global emulator settings (games without their own configuration).
    pub emulator: EmulatorSettings,
    /// Network use; everything is off by default.
    pub network: Network,
    /// The debugging options of "Run with..." last used, per game key.
    pub run_options: BTreeMap<String, RunOptions>,
    /// How each game's last run ended, per game key.
    pub last_runs: BTreeMap<String, LastRun>,
}

/// A game removed from the library: matched by title id when it has one
/// (every copy of the game), else by its path.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct HiddenGame {
    pub path: PathBuf,
    /// Empty when the game has none.
    pub title_id: String,
    /// For the list in the settings.
    pub title: String,
}

impl HiddenGame {
    pub fn of(game: &Game) -> Self {
        Self {
            path: game.source.path().to_path_buf(),
            title_id: game.title_id.clone().unwrap_or_default(),
            title: game.title.clone(),
        }
    }

    pub fn matches(&self, game: &Game) -> bool {
        let id = self.title_id.trim();
        if !id.is_empty() && game.title_id.as_deref().is_some_and(|t| t.trim().eq_ignore_ascii_case(id)) {
            return true;
        }
        self.path == game.source.path()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Network {
    /// Check for a newer version when the launcher starts.
    pub check_updates_on_startup: bool,
    /// Update feed (JSON with `tag`/`tag_name` and an https `html_url`).
    pub update_feed_url: String,
    /// Allow browsing the online cheat collection.
    pub online_cheats: bool,
}

/// The project's latest release (`tag_name`, `html_url`).
pub const DEFAULT_UPDATE_FEED: &str = "https://api.github.com/repos/keklick1337/rps5e/releases/latest";

impl Default for Network {
    fn default() -> Self {
        Self { check_updates_on_startup: false, update_feed_url: DEFAULT_UPDATE_FEED.into(), online_cheats: false }
    }
}

impl Default for GuiSettings {
    fn default() -> Self {
        Self {
            default_library: true,
            library_folders: Vec::new(),
            extra_games: Vec::new(),
            hidden_games: Vec::new(),
            emulator_path: String::new(),
            theme: Theme::System,
            view: LibraryView::Grid,
            emulator: EmulatorSettings::default(),
            network: Network::default(),
            run_options: BTreeMap::new(),
            last_runs: BTreeMap::new(),
        }
    }
}

/// `~/.config/rps5e` (`$RPS5E_CONFIG_DIR`, `$XDG_CONFIG_HOME/rps5e`).
pub fn config_dir() -> Option<PathBuf> {
    ps5input::mapping::config_dir()
}

/// The library folder named by `$GAMES_DIR`, if set.
pub fn default_library_dir() -> Option<PathBuf> {
    default_library_dir_from(std::env::var_os(GAMES_DIR_ENV))
}

/// No built-in folder: without `$GAMES_DIR` the library has only the
/// folders and games added to it.
fn default_library_dir_from(games: Option<std::ffi::OsString>) -> Option<PathBuf> {
    games.filter(|d| !d.is_empty()).map(|d| crate::config::expand_home(&d.to_string_lossy()))
}

/// One row of the library folder list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FolderRow {
    pub path: PathBuf,
    pub enabled: bool,
    /// Named by `$GAMES_DIR` (switched by `default_library`, not removable).
    pub from_env: bool,
}

impl GuiSettings {
    pub fn path() -> Option<PathBuf> {
        config_dir().map(|d| d.join(FILE_NAME))
    }

    pub fn parse(text: &str) -> Result<Self, String> {
        let table: toml::Table = text.parse().map_err(|e: toml::de::Error| e.to_string())?;
        Self::deserialize(table).map_err(|e| e.to_string())
    }

    pub fn to_toml(&self) -> String {
        toml::to_string_pretty(self).unwrap_or_default()
    }

    /// Load from `path`; defaults when it does not exist.
    pub fn load_from(path: &Path) -> Result<Self, String> {
        match std::fs::read_to_string(path) {
            Ok(t) => Self::parse(&t).map_err(|e| format!("{}: {e}", path.display())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(format!("{}: {e}", path.display())),
        }
    }

    pub fn save_to(&self, path: &Path) -> Result<(), String> {
        write_atomic(path, &self.to_toml())
    }

    /// Folders the library scans: the enabled ones, the `$GAMES_DIR` one
    /// first.
    pub fn library_roots(&self) -> Vec<PathBuf> {
        self.folder_rows().into_iter().filter(|r| r.enabled).map(|r| r.path).collect()
    }

    /// Every library folder, enabled or not: `$GAMES_DIR` first, then the
    /// added folders (a folder listed twice shows once).
    pub fn folder_rows(&self) -> Vec<FolderRow> {
        self.folder_rows_with(default_library_dir())
    }

    fn folder_rows_with(&self, env_dir: Option<PathBuf>) -> Vec<FolderRow> {
        let mut out: Vec<FolderRow> = Vec::new();
        if let Some(d) = env_dir {
            out.push(FolderRow { path: d, enabled: self.default_library, from_env: true });
        }
        for f in &self.library_folders {
            match out.iter_mut().find(|r| r.path == f.path) {
                // An added folder that is also `$GAMES_DIR` is scanned when
                // either switch is on.
                Some(r) => r.enabled |= f.enabled,
                None => out.push(FolderRow { path: f.path.clone(), enabled: f.enabled, from_env: false }),
            }
        }
        out
    }

    /// Add library folder `dir`, enabled (an entry already there is turned
    /// on); false when it was already scanned.
    pub fn add_library_dir(&mut self, dir: PathBuf) -> bool {
        if default_library_dir().as_deref() == Some(dir.as_path()) && !self.default_library {
            self.default_library = true;
            return true;
        }
        if self.library_roots().contains(&dir) {
            return false;
        }
        match self.library_folders.iter_mut().find(|f| f.path == dir) {
            Some(f) => f.enabled = true,
            None => self.library_folders.push(LibraryFolder::new(dir)),
        }
        true
    }

    /// Scan library folder `dir` or not (it stays in the list); false when
    /// it is no library folder or nothing changed.
    pub fn set_folder_enabled(&mut self, dir: &Path, on: bool) -> bool {
        self.set_folder_enabled_with(dir, on, default_library_dir().as_deref())
    }

    fn set_folder_enabled_with(&mut self, dir: &Path, on: bool, env_dir: Option<&Path>) -> bool {
        let mut changed = false;
        if env_dir == Some(dir) && self.default_library != on {
            self.default_library = on;
            changed = true;
        }
        for f in self.library_folders.iter_mut().filter(|f| f.path == dir && f.enabled != on) {
            f.enabled = on;
            changed = true;
        }
        changed
    }

    pub fn is_hidden(&self, game: &Game) -> bool {
        self.hidden_games.iter().any(|h| h.matches(game))
    }

    /// Remove `game` from the library (until [`Self::unhide`]); false when
    /// it already was.
    pub fn hide(&mut self, game: &Game) -> bool {
        if self.is_hidden(game) {
            return false;
        }
        self.hidden_games.push(HiddenGame::of(game));
        true
    }

    /// Show the games of hidden entry `index` again.
    pub fn unhide(&mut self, index: usize) -> Option<HiddenGame> {
        (index < self.hidden_games.len()).then(|| self.hidden_games.remove(index))
    }

    /// Show `game` again (it is added by hand again).
    pub fn unhide_game(&mut self, game: &Game) {
        self.hidden_games.retain(|h| !h.matches(game));
    }

    /// The games the library shows: those not hidden.
    pub fn visible(&self, games: Vec<Game>) -> Vec<Game> {
        games.into_iter().filter(|g| !self.is_hidden(g)).collect()
    }

    /// Remove library folder `dir` from the list; false when it was not
    /// one. The `$GAMES_DIR` folder is turned off instead.
    pub fn remove_library_dir(&mut self, dir: &Path) -> bool {
        let before = self.library_folders.len();
        self.library_folders.retain(|f| f.path != dir);
        if self.default_library && default_library_dir().as_deref() == Some(dir) {
            self.default_library = false;
            return true;
        }
        self.library_folders.len() != before
    }

    /// The emulator binary to run.
    pub fn emulator_binary(&self) -> PathBuf {
        if !self.emulator_path.trim().is_empty() {
            return crate::config::expand_home(self.emulator_path.trim());
        }
        let exe = std::env::current_exe().ok();
        // Running as the emulator itself (`rps5e-cli` with no command).
        if let Some(e) = exe.as_ref().filter(|e| e.file_stem().is_some_and(|n| n.to_string_lossy().starts_with("rps5e"))) {
            return e.clone();
        }
        let name = if cfg!(windows) { "rps5e-cli.exe" } else { "rps5e-cli" };
        exe.and_then(|e| e.parent().map(|d| d.join(name))).filter(|p| p.is_file()).unwrap_or_else(|| PathBuf::from(name))
    }
}

/// Write via a temporary file + rename, creating the directory.
pub fn write_atomic(path: &Path, text: &str) -> Result<(), String> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    std::fs::write(&tmp, text).map_err(|e| format!("{}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("{}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn folders(s: &GuiSettings) -> Vec<(PathBuf, bool)> {
        s.library_folders.iter().map(|f| (f.path.clone(), f.enabled)).collect()
    }

    #[test]
    fn round_trip_and_partial_files() {
        let mut s = GuiSettings::default();
        s.library_folders.push(LibraryFolder::new("/games"));
        s.library_folders.push(LibraryFolder { path: "/usb".into(), enabled: false });
        s.emulator.present_mode = "fifo".into();
        s.emulator.mute_audio = true;
        s.theme = Theme::Dark;
        s.network.online_cheats = true;
        s.last_runs.insert("/games/A".into(), LastRun { ended: 5, seconds: 60, result: "exit code 3".into(), ok: false });
        let mut o = RunOptions::default();
        o.gpu_profile = 2;
        s.run_options.insert("/games/A".into(), o);
        assert_eq!(GuiSettings::parse(&s.to_toml()).unwrap(), s);
        // Missing keys take defaults; unknown sections are ignored.
        let p = GuiSettings::parse("[emulator]\nfullscreen = true\n[graphics]\nfullscreen = false\n").unwrap();
        assert!(p.emulator.fullscreen);
        assert_eq!(p.emulator.present_mode, "mailbox");
        assert_eq!(p.emulator.console_language, 1);
        assert!(p.default_library && !p.network.check_updates_on_startup);
        assert!(GuiSettings::parse("library_folders = 5").is_err());
    }

    #[test]
    fn folders_enable_disable_without_removal() {
        let mut s = GuiSettings::default();
        s.default_library = false;
        assert!(s.add_library_dir("/a".into()));
        assert!(s.add_library_dir("/b".into()));
        assert!(!s.add_library_dir("/a".into()), "already scanned");
        assert_eq!(s.library_roots(), [PathBuf::from("/a"), PathBuf::from("/b")]);
        // Off: kept in the list, not scanned.
        assert!(s.set_folder_enabled(Path::new("/a"), false));
        assert!(!s.set_folder_enabled(Path::new("/a"), false), "no change");
        assert!(!s.set_folder_enabled(Path::new("/zzz"), false), "not a library folder");
        assert_eq!(s.library_roots(), [PathBuf::from("/b")]);
        assert_eq!(folders(&s), [(PathBuf::from("/a"), false), ("/b".into(), true)]);
        // Adding a disabled folder again turns it back on, in its place.
        assert!(s.add_library_dir("/a".into()));
        assert_eq!(folders(&s), [(PathBuf::from("/a"), true), ("/b".into(), true)]);
        s.set_folder_enabled(Path::new("/b"), false);
        // The switch survives saving.
        let l = GuiSettings::parse(&s.to_toml()).unwrap();
        assert_eq!(l.library_roots(), [PathBuf::from("/a")]);
        assert_eq!(folders(&l), folders(&s));
        // Removal drops the entry, enabled or not.
        assert!(s.remove_library_dir(Path::new("/b")));
        assert_eq!(folders(&s), [(PathBuf::from("/a"), true)]);
    }

    #[test]
    fn games_dir_folder_row_and_switch() {
        let mut s = GuiSettings::default();
        s.library_folders = vec![LibraryFolder::new("/env"), LibraryFolder { path: "/x".into(), enabled: false }];
        s.default_library = false;
        let rows = s.folder_rows_with(Some("/env".into()));
        assert_eq!(
            rows,
            [
                FolderRow { path: "/env".into(), enabled: true, from_env: true },
                FolderRow { path: "/x".into(), enabled: false, from_env: false },
            ]
        );
        let mut s = GuiSettings::default();
        assert!(s.set_folder_enabled_with(Path::new("/env"), false, Some(Path::new("/env"))));
        assert!(!s.default_library);
        assert_eq!(s.folder_rows_with(Some("/env".into())), [FolderRow { path: "/env".into(), enabled: false, from_env: true }]);
        assert!(s.set_folder_enabled_with(Path::new("/env"), true, Some(Path::new("/env"))));
        assert!(s.default_library);
        assert!(s.folder_rows_with(None).is_empty());
    }

    #[test]
    fn default_library_folder() {
        assert_eq!(default_library_dir_from(Some("/srv/games".into())), Some(PathBuf::from("/srv/games")));
        assert_eq!(default_library_dir_from(Some("".into())), None);
        assert_eq!(default_library_dir_from(None), None);
        let mut s = GuiSettings::default();
        s.default_library = false;
        s.library_folders = vec![LibraryFolder::new("/a"), LibraryFolder::new("/a"), LibraryFolder::new("/b")];
        assert_eq!(s.library_roots(), [PathBuf::from("/a"), PathBuf::from("/b")]);
    }

    fn game(path: &str, id: Option<&str>) -> Game {
        use crate::library::GameSource;
        Game {
            source: GameSource::Folder(path.into()),
            dir: path.into(),
            title: format!("T {path}"),
            title_id: id.map(str::to_string),
            ..Default::default()
        }
    }

    #[test]
    fn hide_and_unhide_persist() {
        let p = crate::test_dir().join(format!("hidden-{}.toml", std::process::id()));
        let a = game("/g/a/PPSA00001", Some("PPSA00001"));
        let a_copy = game("/g/other/PPSA00001-copy", Some("ppsa00001"));
        let b = game("/g/b/homebrew", None);
        let c = game("/g/c/PPSA00003", Some("PPSA00003"));
        let mut s = GuiSettings::default();
        assert!(s.hide(&a) && !s.hide(&a));
        assert!(s.hide(&b));
        s.save_to(&p).unwrap();
        let mut l = GuiSettings::load_from(&p).unwrap();
        assert_eq!(l, s);
        assert_eq!(l.hidden_games[0], HiddenGame { path: "/g/a/PPSA00001".into(), title_id: "PPSA00001".into(), title: "T /g/a/PPSA00001".into() });
        // A title id hides every copy; without one only that path.
        assert!(l.is_hidden(&a) && l.is_hidden(&a_copy) && l.is_hidden(&b) && !l.is_hidden(&c));
        assert!(!l.is_hidden(&game("/g/b/homebrew2", None)));
        let shown = l.visible(vec![a.clone(), b.clone(), c.clone(), a_copy.clone()]);
        assert_eq!(shown, [c.clone()]);
        // Restore one from the list, the other by adding it again.
        assert_eq!(l.unhide(0).map(|h| h.title_id), Some("PPSA00001".into()));
        assert!(l.unhide(5).is_none());
        l.save_to(&p).unwrap();
        let mut l = GuiSettings::load_from(&p).unwrap();
        assert!(!l.is_hidden(&a) && l.is_hidden(&b));
        l.unhide_game(&b);
        l.save_to(&p).unwrap();
        assert!(GuiSettings::load_from(&p).unwrap().hidden_games.is_empty());
        // Files without the list load with none hidden.
        assert!(GuiSettings::parse("theme = \"dark\"\n").unwrap().hidden_games.is_empty());
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn remove_library_folder() {
        let mut s = GuiSettings::default();
        s.library_folders = vec![LibraryFolder::new("/a"), LibraryFolder::new("/b")];
        assert!(s.remove_library_dir(Path::new("/a")));
        assert!(!s.remove_library_dir(Path::new("/zzz")));
        assert_eq!(folders(&s), [(PathBuf::from("/b"), true)]);
    }

    #[test]
    fn load_missing_is_default() {
        let p = crate::test_dir().join(format!("missing-{}.toml", std::process::id()));
        assert_eq!(GuiSettings::load_from(&p).unwrap(), GuiSettings::default());
        let mut s = GuiSettings::default();
        s.emulator_path = "/x/rps5e-cli".into();
        s.save_to(&p).unwrap();
        assert_eq!(GuiSettings::load_from(&p).unwrap(), s);
        let _ = std::fs::remove_file(&p);
    }
}
