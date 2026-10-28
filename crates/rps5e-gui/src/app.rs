// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The launcher's state and actions: the game library (scan, folders,
//! removal and undo), runs of the emulator as a child process (play, run
//! with debugging options, shader compiles, package installs), the
//! dialogs and the frame that lays out the pages (sidebar, library,
//! settings, logs, about). The pages themselves are drawn in
//! `library_ui`, `settings_ui`, `logs_ui` and `pages`.

use crate::compat::{CompatibilityDb, GameStatus};
use crate::config::{EmulatorSettings, GameConfig, GameConfigs};
use crate::dialogs::{CheatsDialog, GameConfigDialog, TrophyDialog, TrophyOverview};
use crate::input_store::{InputStore, Target};
use crate::input_ui::BindingEditor;
use crate::launcher::{self, LevelFilter, LineKind, Log, Running, SharedLog};
use crate::library::{self, Game};
use crate::run_options::{self, RunOptions};
use crate::sections::{self, LibrarySnapshot, Removal, Section};
use crate::settings::{self, GuiSettings, LastRun, Theme};
use crate::updates::UpdateResult;
use eframe::egui::{self, TextureHandle};
use ps5input::mapping::PadInput;
use ps5input::sdl::{DeviceInfo, SdlGamepads};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver};
use std::time::Duration;

/// The launcher's pages (sidebar).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Page {
    Library,
    Settings,
    Logs,
    About,
}

/// Tabs of the settings page.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Tab {
    Library,
    Emulator,
    Input,
    Interface,
}

pub(crate) enum ScanMsg {
    Games(Vec<Game>),
    Icon(PathBuf, u32, u32, Vec<u8>),
    Size(String, u64),
}

/// The order of the library.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Column {
    Name,
    Serial,
    Version,
    Size,
    Status,
    LastRun,
}

impl Column {
    pub const ALL: [Column; 6] = [Column::Name, Column::Serial, Column::Version, Column::Size, Column::Status, Column::LastRun];

    pub fn label(self) -> &'static str {
        match self {
            Column::Name => "Name",
            Column::Serial => "Title ID",
            Column::Version => "Version",
            Column::Size => "Size",
            Column::Status => "Status",
            Column::LastRun => "Last played",
        }
    }
}

/// What a file dialog picks.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum FilePurpose {
    AddLibraryFolder,
    AddGameFolder,
    AddGame,
    /// A package to extract into the data folder's `games/`.
    InstallPackage,
    ImportConfigs,
    ExportConfig(String),
    ImportCheat,
    /// A game's portable shader cache (the game's key).
    ExportShaderCache(String),
    ImportShaderCache(String),
}

/// An action waiting for confirmation.
pub(crate) enum ConfirmAction {
    RemoveSaveData(Vec<PathBuf>),
    ClearGameConfig(String),
    ImportConfigs(Vec<(String, GameConfig)>),
    RemoveFromLibrary(Removal),
    ClearAllShaderCaches,
}

impl ConfirmAction {
    /// The confirming button.
    pub(crate) fn yes_label(&self) -> &'static str {
        match self {
            ConfirmAction::RemoveSaveData(_) => "Delete",
            ConfirmAction::RemoveFromLibrary(_) => "Remove",
            ConfirmAction::ClearAllShaderCaches => "Clear",
            _ => "Yes",
        }
    }
}

pub(crate) struct Confirm {
    pub title: String,
    pub text: String,
    pub action: ConfirmAction,
}

/// What the running child process is.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum RunKind {
    /// A game (its key).
    Game(String),
    /// `install-pkg` of a package (the package title) into the library
    /// folder it lands in.
    Install(String, PathBuf),
    /// `precompile` (`--clear`: recompile) of a game (its title).
    Recompile(String),
    /// `shader-cache export|import` of a game (its title).
    ShaderCache(String),
}

impl RunKind {
    pub fn label(&self) -> &'static str {
        match self {
            RunKind::Game(_) => "Game",
            RunKind::Install(..) => "Package install",
            RunKind::Recompile(_) => "Shader compile",
            RunKind::ShaderCache(_) => "Shader cache",
        }
    }
}

/// The "Run with..." dialog of a game.
pub(crate) struct RunDialog {
    pub key: String,
    pub options: RunOptions,
}

/// State of the log page.
#[derive(Default)]
pub(crate) struct LogView {
    pub level: LevelFilter,
    pub query: String,
    /// Stick to the newest line.
    pub follow: bool,
    /// A saved run log shown instead of the live output.
    pub file: Option<(PathBuf, Log)>,
}

pub struct App {
    pub(crate) settings: GuiSettings,
    pub(crate) settings_path: Option<PathBuf>,
    pub(crate) config_dir: Option<PathBuf>,
    pub(crate) status: String,
    pub(crate) configs: GameConfigs,
    pub(crate) compat: CompatibilityDb,
    applied_theme: Option<Theme>,
    styled: bool,

    pub(crate) page: Page,
    pub(crate) games: Vec<Game>,
    pub(crate) scan_rx: Option<Receiver<ScanMsg>>,
    pub(crate) icons: HashMap<PathBuf, TextureHandle>,
    /// Background pictures of the games shown in the details (None: the
    /// game has none), by game key.
    pub(crate) banners: HashMap<String, Option<TextureHandle>>,
    pub(crate) banner_rx: Option<Receiver<(String, Option<(u32, u32, Vec<u8>)>)>>,
    pub(crate) sizes: HashMap<String, u64>,
    pub(crate) selected: Option<String>,
    pub(crate) filter: String,
    pub(crate) sort: (Column, bool),
    pub(crate) comment_edit: HashMap<String, String>,

    pub(crate) log: SharedLog,
    pub(crate) running: Option<(Running, RunKind)>,
    pub(crate) log_view: LogView,
    pkg_probe: Option<Receiver<bool>>,
    pub(crate) pkg_install: Option<bool>,

    pub(crate) tab: Tab,
    pub(crate) input: Option<InputStore>,
    pub(crate) input_error: Option<String>,
    pub(crate) bindings: BindingEditor,
    pub(crate) sdl: Option<SdlGamepads>,
    pub(crate) sdl_error: Option<String>,
    pub(crate) devices: Vec<DeviceInfo>,
    pub(crate) preview_device: Option<u32>,
    lightbar_shown: Option<([u8; 3], usize)>,
    pub(crate) new_dir: String,

    pub(crate) file_dialog: Option<(egui_file::FileDialog, FilePurpose)>,
    pub(crate) confirm: Option<Confirm>,
    /// The last removal from the library (status text, settings before).
    pub(crate) undo: Option<(String, LibrarySnapshot)>,
    pub(crate) game_config: Option<GameConfigDialog>,
    pub(crate) cheats: Option<CheatsDialog>,
    pub(crate) trophies: Option<TrophyDialog>,
    pub(crate) trophy_overview: Option<TrophyOverview>,
    pub(crate) run_dialog: Option<RunDialog>,
    pub(crate) update_rx: Option<Receiver<UpdateResult>>,
    pub(crate) update_result: Option<UpdateResult>,
    /// The launcher's graphics adapter (for issue reports).
    gpu_name: Option<String>,
    /// The game whose last run failed (its key and how it ended): offers
    /// an issue report.
    crashed: Option<(String, String)>,
    /// Whether the crash window is shown.
    crash_prompt: bool,
    /// Text to put on the clipboard at the end of the frame.
    pub(crate) copy_text: Option<String>,
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>, options: crate::Options) -> Self {
        let settings_path = GuiSettings::path();
        let (settings, mut status) = match settings_path.as_deref().map(GuiSettings::load_from) {
            Some(Ok(s)) => (s, String::new()),
            Some(Err(e)) => (GuiSettings::default(), format!("settings: {e}")),
            None => (GuiSettings::default(), "no configuration directory ($HOME unset)".into()),
        };
        let config_dir = settings::config_dir();
        let configs = match config_dir.as_deref().map(GameConfigs::load) {
            Some(Ok(c)) => c,
            Some(Err(e)) => {
                status = format!("game configurations: {e}");
                GameConfigs::default()
            }
            None => GameConfigs::default(),
        };
        let compat = match config_dir.as_deref().map(CompatibilityDb::load) {
            Some(Ok(c)) => c,
            Some(Err(e)) => {
                log::warn!("{e}");
                CompatibilityDb::default()
            }
            None => CompatibilityDb::default(),
        };
        let mut app = Self::with(settings, settings_path, config_dir, status, configs, compat);
        app.gpu_name = cc.gl.as_ref().map(|gl| {
            use eframe::glow::HasContext;
            // SAFETY: a string query on the current context.
            unsafe { gl.get_parameter_string(eframe::glow::RENDERER) }
        });
        app.rescan();
        app.probe_package_support();
        if app.settings.network.check_updates_on_startup && !app.settings.network.update_feed_url.trim().is_empty() {
            app.check_updates();
        }
        if options.open_settings {
            app.open_input_settings(Target::Global);
        }
        app
    }

    /// The launcher state of loaded settings, without scanning.
    pub(crate) fn with(
        settings: GuiSettings,
        settings_path: Option<PathBuf>,
        config_dir: Option<PathBuf>,
        status: String,
        configs: GameConfigs,
        compat: CompatibilityDb,
    ) -> Self {
        Self {
            settings,
            settings_path,
            config_dir,
            status,
            configs,
            compat,
            applied_theme: None,
            styled: false,
            page: Page::Library,
            games: Vec::new(),
            scan_rx: None,
            icons: HashMap::new(),
            banners: HashMap::new(),
            banner_rx: None,
            sizes: HashMap::new(),
            selected: None,
            filter: String::new(),
            sort: (Column::Name, true),
            comment_edit: HashMap::new(),
            log: SharedLog::default(),
            running: None,
            log_view: LogView { follow: true, ..Default::default() },
            pkg_probe: None,
            pkg_install: None,
            tab: Tab::Library,
            input: None,
            input_error: None,
            bindings: BindingEditor::default(),
            sdl: None,
            sdl_error: None,
            devices: Vec::new(),
            preview_device: None,
            lightbar_shown: None,
            new_dir: String::new(),
            file_dialog: None,
            confirm: None,
            undo: None,
            game_config: None,
            cheats: None,
            trophies: None,
            trophy_overview: None,
            run_dialog: None,
            update_rx: None,
            update_result: None,
            gpu_name: None,
            crashed: None,
            crash_prompt: false,
            copy_text: None,
        }
    }

    pub(crate) fn save_settings(&mut self) {
        if let Some(p) = &self.settings_path {
            if let Err(e) = self.settings.save_to(p) {
                self.status = format!("Cannot save settings: {e}");
            }
        }
    }

    pub(crate) fn save_configs(&mut self) {
        let Some(dir) = self.config_dir.clone() else { return };
        if let Err(e) = self.configs.save(&dir) {
            self.status = format!("Cannot save game configurations: {e}");
        }
    }

    pub(crate) fn rescan(&mut self) {
        let roots = self.settings.library_roots();
        let extra = self.settings.extra_games.clone();
        let (tx, rx) = channel();
        self.scan_rx = Some(rx);
        let have: Vec<PathBuf> = self.icons.keys().cloned().collect();
        let known: Vec<String> = self.sizes.keys().cloned().collect();
        let _ = std::thread::Builder::new().name("rps5e-gui-scan".into()).spawn(move || {
            let (games, package_icons) = library::scan_with_icons(&roots, &extra);
            let icons: Vec<PathBuf> = games
                .iter()
                .filter(|g| !g.source.is_package())
                .filter_map(|g| g.icon.clone())
                .filter(|p| !have.contains(p))
                .collect();
            let sizes: Vec<(String, PathBuf)> =
                games.iter().filter(|g| !known.contains(&g.key())).map(|g| (g.key(), g.source.path().to_path_buf())).collect();
            if tx.send(ScanMsg::Games(games)).is_err() {
                return;
            }
            for (name, bytes) in package_icons {
                if have.contains(&name) {
                    continue;
                }
                if let Some((w, h, rgba)) = library::decode_png_rgba(&bytes[..]) {
                    if tx.send(ScanMsg::Icon(name, w, h, rgba)).is_err() {
                        return;
                    }
                }
            }
            for p in icons {
                match library::load_png_rgba(&p) {
                    Some((w, h, rgba)) => {
                        if tx.send(ScanMsg::Icon(p, w, h, rgba)).is_err() {
                            return;
                        }
                    }
                    None => log::warn!("cannot decode {}", p.display()),
                }
            }
            for (key, path) in sizes {
                if tx.send(ScanMsg::Size(key, library::disk_size(&path))).is_err() {
                    return;
                }
            }
        });
    }

    fn poll_scan(&mut self, ctx: &egui::Context) {
        let Some(rx) = &self.scan_rx else { return };
        let mut done = false;
        loop {
            match rx.try_recv() {
                Ok(ScanMsg::Games(g)) => {
                    let found = g.len();
                    self.games = self.settings.visible(g);
                    let hidden = found - self.games.len();
                    self.status = match hidden {
                        0 => format!("{found} game(s) found"),
                        n => format!("{} game(s) found, {n} hidden", self.games.len()),
                    };
                    if self.selected.as_ref().is_some_and(|k| self.game(k).is_none()) {
                        self.selected = None;
                    }
                }
                Ok(ScanMsg::Icon(p, w, h, rgba)) => {
                    let img = egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &rgba);
                    let tex = ctx.load_texture(p.display().to_string(), img, egui::TextureOptions::LINEAR);
                    self.icons.insert(p, tex);
                }
                Ok(ScanMsg::Size(k, s)) => {
                    self.sizes.insert(k, s);
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    done = true;
                    break;
                }
            }
        }
        if done {
            self.scan_rx = None;
            ctx.request_repaint();
        } else {
            ctx.request_repaint_after(Duration::from_millis(50));
        }
    }

    /// The background picture of `g` for its details: loaded (shrunk) in
    /// the background the first time it is asked for.
    pub(crate) fn banner(&mut self, g: &Game) -> Option<TextureHandle> {
        let key = g.key();
        if let Some(b) = self.banners.get(&key) {
            return b.clone();
        }
        if self.banner_rx.is_none() && !g.source.is_package() {
            let (tx, rx) = channel();
            self.banner_rx = Some(rx);
            let dir = g.dir.clone();
            let _ = std::thread::Builder::new().name("rps5e-gui-banner".into()).spawn(move || {
                let _ = tx.send((key, library::load_banner(&dir, 720)));
            });
        } else if g.source.is_package() {
            self.banners.insert(key, None);
        }
        None
    }

    fn poll_banner(&mut self, ctx: &egui::Context) {
        let Some(rx) = &self.banner_rx else { return };
        match rx.try_recv() {
            Ok((key, img)) => {
                let tex = img.map(|(w, h, rgba)| {
                    let img = egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &rgba);
                    ctx.load_texture(format!("banner {key}"), img, egui::TextureOptions::LINEAR)
                });
                self.banners.insert(key, tex);
                self.banner_rx = None;
                ctx.request_repaint();
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => ctx.request_repaint_after(Duration::from_millis(50)),
            Err(std::sync::mpsc::TryRecvError::Disconnected) => self.banner_rx = None,
        }
    }

    /// Whether the emulator can install packages (`install-pkg` command).
    fn probe_package_support(&mut self) {
        let bin = self.settings.emulator_binary();
        let (tx, rx) = channel();
        self.pkg_probe = Some(rx);
        let _ = std::thread::Builder::new().name("rps5e-gui-probe".into()).spawn(move || {
            let out = std::process::Command::new(&bin).arg("help").stdin(std::process::Stdio::null()).output();
            let ok = out.is_ok_and(|o| String::from_utf8_lossy(&o.stdout).contains("install-pkg"));
            let _ = tx.send(ok);
        });
    }

    pub(crate) fn check_updates(&mut self) {
        let url = self.settings.network.update_feed_url.clone();
        let (tx, rx) = channel();
        self.update_rx = Some(rx);
        let _ = std::thread::Builder::new().name("rps5e-gui-update".into()).spawn(move || {
            let _ = tx.send(crate::updates::check(&url));
        });
    }

    fn poll_background(&mut self) {
        if self.pkg_install.is_none() && self.pkg_probe.is_none() {
            // The emulator binary changed: ask the new one.
            self.probe_package_support();
        }
        if let Some(rx) = &self.pkg_probe {
            if let Ok(v) = rx.try_recv() {
                self.pkg_install = Some(v);
                self.pkg_probe = None;
            }
        }
        if let Some(rx) = &self.update_rx {
            if let Ok(r) = rx.try_recv() {
                self.update_result = Some(r);
                self.update_rx = None;
            }
        }
    }

    pub(crate) fn is_running(&self) -> bool {
        self.running.as_ref().is_some_and(|r| r.0.exit.is_none())
    }

    pub(crate) fn running_game(&self) -> Option<&str> {
        match &self.running {
            Some((r, RunKind::Game(k))) if r.exit.is_none() => Some(k),
            _ => None,
        }
    }

    pub(crate) fn game(&self, key: &str) -> Option<&Game> {
        self.games.iter().find(|g| g.key() == key)
    }

    pub(crate) fn selected_game(&self) -> Option<Game> {
        self.selected.as_ref().and_then(|k| self.game(k)).cloned()
    }

    /// The settings `game` runs with.
    pub(crate) fn effective(&self, game: &Game) -> EmulatorSettings {
        self.configs.effective(&game.key(), &self.settings.emulator).clone()
    }

    /// Where the debugging output of `game` goes.
    pub(crate) fn debug_dir(&self, game: &Game) -> PathBuf {
        run_options::debug_dir(&self.effective(game).effective_data_dir(), game.title_id.as_deref().unwrap_or(&game.title))
    }

    /// The executable and environment of a run of `game` (with the
    /// debugging options `opts`).
    pub(crate) fn run_plan(&self, game: &Game, opts: Option<&RunOptions>) -> Result<(PathBuf, Vec<(String, String)>), String> {
        if let Some(state) = game.package.as_ref().filter(|_| game.source.is_package()) {
            if !state.can_run() {
                return Err(format!("{}: {} ({})", game.title, state.label(), state.detail()));
            }
        }
        let cfg = self.effective(game);
        // A runnable package is mounted as /app0 by `run`.
        let exe = if game.source.is_package() { game.source.path().to_path_buf() } else { game.executable(&cfg.elf) };
        let patch = game
            .title_id
            .as_deref()
            .and_then(|id| crate::cheats::plan_path(&cfg.effective_data_dir(), id))
            .filter(|p| p.is_file());
        let input_cfg = self.config_dir.as_deref().and_then(InputStore::global_file);
        let mut env = cfg.env(self.config_dir.as_deref(), input_cfg.as_deref(), patch.as_deref());
        // For the in-game menu: where the game's settings are saved.
        env.push((crate::ingame::GAME_KEY_ENV.into(), game.key()));
        env.push((crate::ingame::GAME_TITLE_ENV.into(), game.title.clone()));
        env.push((crate::ingame::FROM_LAUNCHER_ENV.into(), "1".into()));
        if let Some(o) = opts {
            o.validate()?;
            o.apply(&mut env, &self.debug_dir(game));
        }
        Ok((exe, env))
    }

    /// The shell line that runs `game` like the launcher does.
    pub(crate) fn command_line(&self, game: &Game, opts: Option<&RunOptions>) -> Result<String, String> {
        let (exe, env) = self.run_plan(game, opts)?;
        Ok(launcher::launch_command_line(&self.settings.emulator_binary(), &exe, &env))
    }

    pub(crate) fn launch(&mut self, game: &Game) {
        self.launch_with(game, None);
    }

    /// Run `game`, with the debugging options `opts` ("Run with...").
    pub(crate) fn launch_with(&mut self, game: &Game, opts: Option<&RunOptions>) {
        if self.is_running() {
            self.status = "Another program is already running".into();
            return;
        }
        let (exe, env) = match self.run_plan(game, opts) {
            Ok(p) => p,
            Err(e) => {
                self.status = e;
                return;
            }
        };
        if !exe.is_file() {
            self.status = format!("{} not found", exe.display());
            return;
        }
        if let Some(o) = opts {
            if let Err(e) = o.prepare(&self.debug_dir(game)) {
                self.status = format!("Cannot prepare the debug output: {e}");
                return;
            }
        }
        let bin = self.settings.emulator_binary();
        let log_file = launcher::new_log_file(&launcher::logs_dir(), game.title_id.as_deref().unwrap_or(&game.title));
        match launcher::launch(&bin, &exe, &game.title, &env, &self.log, log_file.as_deref()) {
            Ok(r) => {
                self.status = match opts.map(RunOptions::summary).filter(|s| !s.is_empty()) {
                    Some(s) => format!("Running {} ({s})", game.title),
                    None => format!("Running {}", game.title),
                };
                self.running = Some((r, RunKind::Game(game.key())));
                self.crashed = None;
                self.crash_prompt = false;
                self.log_view.follow = true;
                self.log_view.file = None;
            }
            Err(e) => {
                launcher::log_push(&self.log, LineKind::Error, e.clone());
                self.status = e;
            }
        }
    }

    /// Build the game's shader list now (`precompile`), or clear its
    /// shader and pipeline caches and build them again (`clear`:
    /// `precompile --clear`), with its progress in the status bar.
    fn recompile_shaders(&mut self, game: &Game, clear: bool) {
        if self.is_running() {
            self.status = "Another program is already running".into();
            return;
        }
        let Some(id) = game.title_id.clone() else {
            self.status = format!("{}: no title id, so no shader cache", game.title);
            return;
        };
        let bin = self.settings.emulator_binary();
        let mut args = vec![std::ffi::OsString::from("precompile")];
        if clear {
            args.push("--clear".into());
        }
        args.push(std::ffi::OsString::from(&id));
        let cfg = self.effective(game);
        let env = cfg.env(self.config_dir.as_deref(), None, None);
        let what = if clear { "recompile" } else { "compile" };
        match launcher::spawn(&bin, &args, game.source.path(), &format!("{what} shaders of {}", game.title), &env, &self.log, None) {
            Ok(r) => {
                self.status = format!("{} the shaders of {}", if clear { "Recompiling" } else { "Compiling" }, game.title);
                self.running = Some((r, RunKind::Recompile(game.title.clone())));
                self.log_view.follow = true;
            }
            Err(e) => self.status = e,
        }
    }

    /// Reset compiled shaders (`shader-cache clear`) of `game`, or of every
    /// game and the translated-module caches when `None`: the next start or
    /// precompile builds them again.
    pub(crate) fn clear_shader_cache(&mut self, game: Option<&Game>) {
        if self.is_running() {
            self.status = "Another program is already running".into();
            return;
        }
        let mut args = vec![std::ffi::OsString::from("shader-cache"), "clear".into()];
        let (label, cwd, env) = match game {
            Some(g) => {
                let Some(id) = g.title_id.clone() else {
                    self.status = format!("{}: no title id, so no shader cache", g.title);
                    return;
                };
                args.push(id.into());
                (g.title.clone(), g.source.path().to_path_buf(), self.effective(g).env(self.config_dir.as_deref(), None, None))
            }
            None => {
                args.push("--all".into());
                ("All games".to_string(), std::env::temp_dir(), Vec::new())
            }
        };
        let bin = self.settings.emulator_binary();
        match launcher::spawn(&bin, &args, &cwd, &format!("clear the shader cache of {label}"), &env, &self.log, None) {
            Ok(r) => {
                self.status = format!("{label}: clearing the shader cache");
                self.running = Some((r, RunKind::ShaderCache(label)));
                self.log_view.follow = true;
            }
            Err(e) => self.status = e,
        }
    }

    /// Export the game's shader cache to `file`, or merge `file` into it
    /// (`shader-cache export|import`): a portable file other players can
    /// import to skip the first-run compiles.
    fn shader_cache(&mut self, game: &Game, file: &Path, import: bool) {
        if self.is_running() {
            self.status = "Another program is already running".into();
            return;
        }
        let Some(id) = game.title_id.clone() else {
            self.status = format!("{}: no title id, so no shader cache", game.title);
            return;
        };
        let bin = self.settings.emulator_binary();
        let args: Vec<std::ffi::OsString> = if import {
            vec!["shader-cache".into(), "import".into(), file.as_os_str().to_owned(), id.into()]
        } else {
            vec!["shader-cache".into(), "export".into(), id.into(), file.as_os_str().to_owned()]
        };
        let env = self.effective(game).env(self.config_dir.as_deref(), None, None);
        let what = if import { "import the shader cache of" } else { "export the shader cache of" };
        match launcher::spawn(&bin, &args, game.source.path(), &format!("{what} {}", game.title), &env, &self.log, None) {
            Ok(r) => {
                self.status = format!("{}: {} {}", game.title, if import { "importing shader cache from" } else { "exporting shader cache to" }, file.display());
                self.running = Some((r, RunKind::ShaderCache(game.title.clone())));
                self.log_view.follow = true;
            }
            Err(e) => self.status = e,
        }
    }

    /// The progress of a running shader compile: (fraction, line).
    pub(crate) fn recompile_progress(&self) -> Option<(f32, String)> {
        let Some((r, RunKind::Recompile(title))) = &self.running else { return None };
        if r.exit.is_some() {
            return None;
        }
        let log = self.log.lock().unwrap_or_else(|e| e.into_inner());
        let found = log.lines.iter().rev().take(200).find_map(|l| launcher::parse_progress(&l.text));
        Some(match found {
            Some((done, total, rest)) => {
                let frac = if total == 0 { 1.0 } else { done as f32 / total as f32 };
                (frac, format!("Compiling shaders of {title}: {done}/{total}{rest}"))
            }
            None => (0.0, format!("Compiling shaders of {title}: loading the list")),
        })
    }

    /// Extract package `pkg` into `<data>/games/<TITLE_ID>`
    /// (`install-pkg`); that folder joins the library when it is done.
    pub(crate) fn install_package(&mut self, pkg: &Path, title: &str) {
        if self.is_running() {
            self.status = "Another program is already running".into();
            return;
        }
        if self.pkg_install != Some(true) {
            self.status = "This emulator build cannot install packages (no `install-pkg` command)".into();
            return;
        }
        let bin = self.settings.emulator_binary();
        let data = self.settings.emulator.effective_data_dir();
        let args = [std::ffi::OsString::from("install-pkg"), pkg.as_os_str().to_owned(), "--data-dir".into(), data.as_os_str().to_owned()];
        let env = self.settings.emulator.env(self.config_dir.as_deref(), None, None);
        match launcher::spawn(&bin, &args, pkg, &format!("install {title}"), &env, &self.log, None) {
            Ok(r) => {
                self.status = format!("Installing {title} into {}", data.join("games").display());
                self.running = Some((r, RunKind::Install(title.to_string(), data.join("games"))));
                self.log_view.follow = true;
            }
            Err(e) => self.status = e,
        }
    }

    /// The in-game menu may have changed the game's configuration and input
    /// profile: read them again.
    fn reload_after_game(&mut self) {
        if let Some(dir) = self.config_dir.as_deref() {
            match GameConfigs::load(dir) {
                Ok(c) => self.configs = c,
                Err(e) => self.status = format!("game configurations: {e}"),
            }
        }
        if let Some(target) = self.input.as_ref().filter(|s| !s.dirty).map(|s| s.target.clone()) {
            self.load_input(target);
        }
    }

    pub(crate) fn open_input_settings(&mut self, target: Target) {
        self.page = Page::Settings;
        self.tab = Tab::Input;
        self.load_input(target);
    }

    pub(crate) fn load_input(&mut self, target: Target) {
        self.bindings.capture = None;
        let Some(dir) = self.config_dir.clone() else {
            self.input_error = Some("no configuration directory".into());
            return;
        };
        match InputStore::load(&dir, target) {
            Ok(s) => {
                self.input = Some(s);
                self.input_error = None;
            }
            Err(e) => self.input_error = Some(e),
        }
    }

    pub(crate) fn add_library_dir(&mut self, path: PathBuf) {
        if !path.is_dir() {
            self.status = format!("'{}' is not a folder", path.display());
            return;
        }
        if self.settings.add_library_dir(path.clone()) {
            self.save_settings();
            self.status = format!("Added library folder {}", path.display());
            self.rescan();
        } else {
            self.status = format!("{} is already in the library", path.display());
        }
    }

    /// Scan library folder `dir` or not; it stays in the list.
    pub(crate) fn set_folder_enabled(&mut self, dir: &Path, on: bool) {
        if self.settings.set_folder_enabled(dir, on) {
            self.save_settings();
            self.status = format!("{} {}", if on { "Scanning" } else { "No longer scanning" }, dir.display());
            self.rescan();
        }
    }

    /// Listed games per scanned library folder.
    pub(crate) fn folder_counts(&self) -> HashMap<PathBuf, usize> {
        sections::group(&self.settings, &self.games)
            .into_iter()
            .filter_map(|(s, g)| match s {
                Section::Folder(p) => Some((p, g.len())),
                Section::Added => None,
            })
            .collect()
    }

    /// Ask before removing a game or a section from the library.
    pub(crate) fn ask_remove(&mut self, r: Removal) {
        let p = sections::plan(&self.settings, &self.games, &r);
        self.confirm = Some(Confirm { title: p.title, text: p.text, action: ConfirmAction::RemoveFromLibrary(r) });
    }

    /// Remove a game (hidden until restored) or a section (a library
    /// folder is no longer searched, added games are dropped) from the
    /// library; nothing on disk changes. Undo puts the settings back.
    pub(crate) fn remove_from_library(&mut self, r: Removal) {
        let p = sections::plan(&self.settings, &self.games, &r);
        let before = sections::apply(&mut self.settings, &self.games, &r);
        self.save_settings();
        self.games.retain(|g| !p.keys.contains(&g.key()));
        if self.selected.as_ref().is_some_and(|k| self.game(k).is_none()) {
            self.selected = None;
        }
        let n = p.keys.len();
        let what = match &r {
            Removal::Game(g) => format!("\"{}\"", g.title),
            Removal::Section(Section::Folder(d)) => format!("library folder {} ({n} game{})", d.display(), if n == 1 { "" } else { "s" }),
            Removal::Section(Section::Added) => format!("{n} added game{}", if n == 1 { "" } else { "s" }),
        };
        self.status = format!("Removed {what} from the library");
        self.undo = Some((self.status.clone(), before));
    }

    /// Undo the last removal from the library.
    pub(crate) fn undo_removal(&mut self) {
        let Some((what, before)) = self.undo.take() else { return };
        before.restore(&mut self.settings);
        self.save_settings();
        self.status = format!("Undone: {}", what.to_lowercase());
        self.rescan();
    }

    /// Add one game: a game folder (or its eboot.bin), a package file or an
    /// executable. Shown again if it was removed from the library.
    pub(crate) fn add_game(&mut self, path: PathBuf) {
        let path = if path.is_file() && path.file_name().is_some_and(|n| n.eq_ignore_ascii_case("eboot.bin")) {
            path.parent().map(Path::to_path_buf).unwrap_or(path)
        } else {
            path
        };
        let Some(game) = library::read_entry(&path) else {
            self.status = format!("'{}' is not a game folder (eboot.bin), a package (.pkg, .ffpkg) or an executable (.elf, .self)", path.display());
            return;
        };
        if !self.settings.extra_games.contains(&path) {
            self.settings.extra_games.push(path.clone());
        }
        self.settings.unhide_game(&game);
        self.save_settings();
        self.status = format!("Added {}", game.title);
        self.selected = Some(path.display().to_string());
        self.rescan();
    }

    pub(crate) fn open_file_dialog(&mut self, purpose: FilePurpose) {
        let start = self.settings.library_roots().into_iter().find(|d| d.is_dir());
        let json = |p: &Path| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("json"));
        let mut d = match &purpose {
            FilePurpose::AddLibraryFolder => egui_file::FileDialog::select_folder(start).title("Add library folder (searched for games)"),
            FilePurpose::AddGameFolder => egui_file::FileDialog::select_folder(start).title("Add game folder (with eboot.bin)"),
            FilePurpose::AddGame => egui_file::FileDialog::open_file(start)
                .title("Add game (eboot.bin, .pkg, .ffpkg, .elf, .self)")
                .show_files_filter(Box::new(|p: &Path| library::is_package_file(p) || library::is_executable_file(p))),
            FilePurpose::InstallPackage => egui_file::FileDialog::open_file(start)
                .title("Install package (.pkg, .ffpkg)")
                .show_files_filter(Box::new(library::is_package_file)),
            FilePurpose::ImportConfigs => egui_file::FileDialog::open_file(None).title("Import game configs").show_files_filter(Box::new(json)),
            FilePurpose::ImportCheat => egui_file::FileDialog::open_file(None).title("Import cheat file").show_files_filter(Box::new(json)),
            FilePurpose::ExportConfig(key) => {
                let id = self.configs.games.get(key).map(|g| g.title_id.clone()).unwrap_or_default();
                egui_file::FileDialog::save_file(None).title("Export game config").default_filename(format!("{id}.json"))
            }
            FilePurpose::ExportShaderCache(key) => {
                let id = self.game(key).and_then(|g| g.title_id.clone()).unwrap_or_default();
                egui_file::FileDialog::save_file(None).title("Export shader cache").default_filename(format!("{id}{SHADER_CACHE_SUFFIX}"))
            }
            FilePurpose::ImportShaderCache(_) => egui_file::FileDialog::open_file(None)
                .title("Import shader cache")
                .show_files_filter(Box::new(|p: &Path| p.to_string_lossy().ends_with(SHADER_CACHE_SUFFIX))),
        };
        d.open();
        self.file_dialog = Some((d, purpose));
    }

    fn file_dialog_window(&mut self, ctx: &egui::Context) {
        let Some((d, purpose)) = self.file_dialog.as_mut() else { return };
        let picked = if d.show(ctx).selected() { d.path().map(Path::to_path_buf) } else { None };
        let closed = !d.visible();
        let purpose = purpose.clone();
        if closed {
            self.file_dialog = None;
        }
        let Some(path) = picked else { return };
        self.file_dialog = None;
        match purpose {
            FilePurpose::AddLibraryFolder => self.add_library_dir(path),
            FilePurpose::AddGameFolder | FilePurpose::AddGame => self.add_game(path),
            FilePurpose::InstallPackage => {
                let title = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                self.install_package(&path, &title);
            }
            FilePurpose::ImportConfigs => self.import_configs(&path),
            FilePurpose::ImportCheat => {
                if let Some(c) = self.cheats.as_mut() {
                    c.import_file(&path);
                }
            }
            FilePurpose::ExportConfig(key) => match self.configs.export_json(&key) {
                Ok(text) => match settings::write_atomic(&path, &text) {
                    Ok(()) => self.status = format!("Exported {}", path.display()),
                    Err(e) => self.status = format!("Export failed: {e}"),
                },
                Err(e) => self.status = format!("Export failed: {e}"),
            },
            FilePurpose::ExportShaderCache(ref key) | FilePurpose::ImportShaderCache(ref key) => {
                let import = matches!(purpose, FilePurpose::ImportShaderCache(_));
                if let Some(g) = self.game(key).cloned() {
                    self.shader_cache(&g, &path, import);
                }
            }
        }
    }

    /// Read an import file and ask for confirmation.
    fn import_configs(&mut self, path: &Path) {
        let parsed = std::fs::read_to_string(path).map_err(|e| e.to_string()).and_then(|t| crate::config::parse_import(&t));
        let import = match parsed {
            Ok(i) => i,
            Err(e) => {
                self.status = format!("Import failed: {e}");
                return;
            }
        };
        let mut matched = Vec::new();
        let mut unmatched: Vec<String> = import.0.keys().cloned().collect();
        for g in &self.games {
            let Some(id) = g.title_id.as_deref().map(str::to_ascii_uppercase) else { continue };
            let Some(obj) = import.0.get(&id) else { continue };
            let mut s = self.effective(g);
            if let Err(e) = s.apply_json(obj) {
                self.status = format!("Import failed: Invalid game config for {id}. {e}");
                return;
            }
            matched.push((g.key(), GameConfig { title_id: id.clone(), name: g.title.clone(), settings: s }));
            unmatched.retain(|u| *u != id);
        }
        let mut text = if matched.is_empty() {
            "No matching games found in the library.".to_string()
        } else {
            "Configs will be updated for the following games:\n".to_string()
        };
        for (_, c) in &matched {
            text.push_str(&format!("\n{} - {}", c.title_id, c.name));
        }
        if !unmatched.is_empty() {
            text.push_str(&format!("\nGames not found: {}", unmatched.join(", ")));
        }
        if matched.is_empty() {
            self.status = text.replace('\n', " ");
            return;
        }
        self.confirm = Some(Confirm { title: "Import game configs".into(), text, action: ConfirmAction::ImportConfigs(matched) });
    }

    pub(crate) fn save_data_dirs(&self, game: &Game) -> Vec<PathBuf> {
        let Some(id) = game.title_id.as_deref().filter(|id| ps5input::profile::valid_title_id(id)) else { return Vec::new() };
        let dir = self.effective(game).effective_savedata_root().join(id);
        if dir.is_dir() {
            vec![dir]
        } else {
            Vec::new()
        }
    }

    fn confirm_window(&mut self, ctx: &egui::Context) {
        let Some(c) = &self.confirm else { return };
        let mut answer = None;
        egui::Window::new(&c.title)
            .collapsible(false)
            .resizable(false)
            .default_width(460.0)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                ui.add_space(4.0);
                ui.label(&c.text);
                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    let danger = matches!(c.action, ConfirmAction::RemoveSaveData(_) | ConfirmAction::RemoveFromLibrary(_));
                    let yes = if danger {
                        crate::theme::danger_button(ui, c.action.yes_label(), egui::vec2(96.0, 30.0))
                    } else {
                        crate::theme::primary_button(ui, true, c.action.yes_label(), egui::vec2(96.0, 30.0))
                    };
                    if yes.clicked() {
                        answer = Some(true);
                    }
                    if crate::theme::button(ui, true, "Cancel", egui::vec2(96.0, 30.0)).clicked() {
                        answer = Some(false);
                    }
                });
            });
        let Some(yes) = answer else { return };
        let Some(c) = self.confirm.take() else { return };
        if !yes {
            return;
        }
        match c.action {
            ConfirmAction::RemoveSaveData(dirs) => {
                let failed: Vec<String> =
                    dirs.iter().filter(|d| d.exists() && std::fs::remove_dir_all(d).is_err()).map(|d| d.display().to_string()).collect();
                self.status = if failed.is_empty() { "Save data removed".into() } else { format!("Could not remove: {}", failed.join(", ")) };
            }
            ConfirmAction::ClearAllShaderCaches => self.clear_shader_cache(None),
            ConfirmAction::ClearGameConfig(key) => {
                self.configs.games.remove(&key);
                self.save_configs();
                self.status = "The game uses the global settings now".into();
            }
            ConfirmAction::ImportConfigs(list) => {
                let n = list.len();
                for (k, c) in list {
                    self.configs.games.insert(k, c);
                }
                self.save_configs();
                self.status = format!("Imported {n} game config(s)");
            }
            ConfirmAction::RemoveFromLibrary(r) => self.remove_from_library(r),
        }
    }

    // --- controllers (only while the input settings are shown) -----------

    fn update_controllers(&mut self) {
        if !(self.page == Page::Settings && self.tab == Tab::Input) {
            // Release the devices for the emulator.
            self.sdl = None;
            self.devices.clear();
            self.lightbar_shown = None;
            return;
        }
        if self.sdl.is_none() && self.sdl_error.is_none() {
            match SdlGamepads::new() {
                Ok(s) => self.sdl = Some(s),
                Err(e) => self.sdl_error = Some(e),
            }
        }
        if let Some(s) = self.sdl.as_mut() {
            s.pump();
            self.devices = s.devices();
            if self.preview_device.is_none_or(|id| !self.devices.iter().any(|d| d.id == id)) {
                self.preview_device = self.devices.first().map(|d| d.id);
            }
        }
        // Light bar colour preview on every connected controller.
        let color = self.input.as_ref().map(|s| s.config.features.lightbar_color.unwrap_or(ps5input::sdl::DEFAULT_LIGHTBAR));
        if let (Some(c), Some(s)) = (color, self.sdl.as_mut()) {
            let shown = (c, self.devices.len());
            if self.lightbar_shown != Some(shown) {
                for d in &self.devices {
                    s.set_led(d.id, c);
                }
                self.lightbar_shown = Some(shown);
            }
        }
    }

    /// Feed this frame's input to a pending binding capture.
    fn feed_capture(&mut self, ctx: &egui::Context) {
        let pad_now: Vec<(u32, Vec<PadInput>)> = match &self.sdl {
            Some(s) => self.devices.iter().map(|d| (d.id, s.pressed_inputs(d.id))).collect(),
            None => Vec::new(),
        };
        let mut scratch = ps5input::mapping::Config::default();
        let store = self.input.as_mut();
        let cfg = match store {
            Some(s) => &mut s.config,
            None => &mut scratch,
        };
        if self.bindings.feed(ctx, pad_now, cfg) {
            if let Some(s) = self.input.as_mut() {
                s.dirty = true;
            }
        }
    }

    fn apply_theme(&mut self, ctx: &egui::Context) {
        if !self.styled {
            crate::theme::install(ctx);
            self.styled = true;
        }
        if self.applied_theme == Some(self.settings.theme) {
            return;
        }
        ctx.set_theme(match self.settings.theme {
            Theme::System => egui::ThemePreference::System,
            Theme::Dark => egui::ThemePreference::Dark,
            Theme::Light => egui::ThemePreference::Light,
        });
        self.applied_theme = Some(self.settings.theme);
    }

    /// The compatibility status of `g`.
    pub(crate) fn status_of(&self, g: &Game) -> GameStatus {
        g.title_id.as_deref().and_then(|id| self.compat.find(id)).map(|e| e.status).unwrap_or_default()
    }

    /// The listed games matching the search, in the chosen order.
    pub(crate) fn visible_games(&self) -> Vec<Game> {
        let filter = self.filter.trim().to_lowercase();
        let mut games: Vec<Game> = self
            .games
            .iter()
            .filter(|g| {
                filter.is_empty()
                    || g.title.to_lowercase().contains(&filter)
                    || g.title_id.as_deref().unwrap_or("").to_lowercase().contains(&filter)
            })
            .cloned()
            .collect();
        let (col, asc) = self.sort;
        let last = |g: &Game| self.settings.last_runs.get(&g.key()).map(|r| r.ended).unwrap_or(0);
        games.sort_by(|a, b| {
            let o = match col {
                Column::Name => a.title.to_lowercase().cmp(&b.title.to_lowercase()),
                Column::Serial => a.title_id.cmp(&b.title_id),
                Column::Version => a.version.cmp(&b.version),
                Column::Size => self.sizes.get(&a.key()).cmp(&self.sizes.get(&b.key())),
                Column::Status => self.status_of(a).cmp(&self.status_of(b)),
                // Newest first when ascending.
                Column::LastRun => last(b).cmp(&last(a)),
            };
            let o = o.then_with(|| a.title.to_lowercase().cmp(&b.title.to_lowercase()));
            if asc {
                o
            } else {
                o.reverse()
            }
        });
        games
    }

    pub(crate) fn do_action(&mut self, a: Action) {
        let game = |k: &str| self.game(k).cloned();
        match a {
            Action::Select(k) => self.selected = Some(k),
            Action::Launch(k) => {
                if let Some(g) = game(&k) {
                    self.launch(&g);
                }
            }
            Action::RunWith(k) => {
                let options = self.settings.run_options.get(&k).cloned().unwrap_or_default();
                self.run_dialog = Some(RunDialog { key: k, options });
            }
            Action::CopyCommand(k) => {
                if let Some(g) = game(&k) {
                    let opts = self.settings.run_options.get(&k).cloned();
                    match self.command_line(&g, opts.as_ref()) {
                        Ok(line) => {
                            self.copy_text = Some(line);
                            self.status = "Command line copied".into();
                        }
                        Err(e) => self.status = e,
                    }
                }
            }
            Action::Stop => {
                if let Some(r) = self.running.as_mut() {
                    r.0.stop(&self.log);
                }
            }
            Action::OpenFolder(k) => {
                if let Some(g) = game(&k) {
                    self.open_path(&g.dir);
                }
            }
            Action::Trophies(k) => {
                if let Some(g) = game(&k) {
                    match TrophyDialog::load(self, &g) {
                        Ok(d) => self.trophies = Some(d),
                        Err(e) => self.status = e,
                    }
                }
            }
            Action::TrophyOverview => self.trophy_overview = Some(TrophyOverview::load(self)),
            Action::Cheats(k) => {
                if let Some(g) = game(&k) {
                    self.cheats = Some(CheatsDialog::new(self, &g));
                }
            }
            Action::RemoveSaveData(k) => {
                if let Some(g) = game(&k) {
                    let dirs = self.save_data_dirs(&g);
                    if dirs.is_empty() {
                        self.status = "No save data folder found for this game.".into();
                    } else {
                        let list: Vec<String> = dirs.iter().map(|d| d.display().to_string()).collect();
                        self.confirm = Some(Confirm {
                            title: "Remove save data".into(),
                            text: format!(
                                "Remove save data for \"{}\"?\n\nThis will delete:\n{}\n\nThis cannot be undone.",
                                g.title,
                                list.join("\n")
                            ),
                            action: ConfirmAction::RemoveSaveData(dirs),
                        });
                    }
                }
            }
            Action::Recompile(k) => {
                if let Some(g) = game(&k) {
                    self.recompile_shaders(&g, true);
                }
            }
            Action::Precompile(k) => {
                if let Some(g) = game(&k) {
                    self.recompile_shaders(&g, false);
                }
            }
            Action::ClearShaderCache(k) => {
                if let Some(g) = game(&k) {
                    self.clear_shader_cache(Some(&g));
                }
            }
            Action::ClearAllShaderCaches => {
                self.confirm = Some(Confirm {
                    title: "Clear all shader caches".into(),
                    text: "Delete the compiled shaders and pipelines of every game?\n\nThe shader lists stay: each game builds its shaders again before its next start (this takes a while once).".into(),
                    action: ConfirmAction::ClearAllShaderCaches,
                });
            }
            Action::ExportShaderCache(k) => self.open_file_dialog(FilePurpose::ExportShaderCache(k)),
            Action::ImportShaderCache(k) => self.open_file_dialog(FilePurpose::ImportShaderCache(k)),
            Action::Install(k) => {
                if let Some(g) = game(&k) {
                    self.install_package(g.source.path(), &g.title);
                }
            }
            Action::ReportIssue(k) => {
                if let Some(g) = game(&k) {
                    self.report_issue(&g);
                }
            }
            Action::OpenLogs => {
                let dir = launcher::logs_dir();
                let _ = std::fs::create_dir_all(&dir);
                self.open_path(&dir);
            }
            Action::EditConfig(k) => {
                if let Some(g) = game(&k) {
                    self.game_config = Some(GameConfigDialog::new(self, &g));
                }
            }
            Action::ClearConfig(k) => {
                self.confirm = Some(Confirm {
                    title: "Clear game config".into(),
                    text: "Clear this game's config and use global settings?".into(),
                    action: ConfirmAction::ClearGameConfig(k),
                });
            }
            Action::InputProfile(k) => {
                if let Some(g) = game(&k) {
                    if let Some(id) = g.title_id {
                        self.open_input_settings(Target::Game { title_id: id, title: g.title });
                    }
                }
            }
            Action::SetStatus(id, s) => {
                if let Err(e) = self.compat.set_status(&id, s) {
                    self.status = e;
                }
            }
            Action::SetComment(id, c) => {
                if let Err(e) = self.compat.set_comment(&id, &c) {
                    self.status = e;
                }
            }
            Action::RemoveGame(k) => {
                if let Some(g) = game(&k) {
                    self.ask_remove(Removal::Game(g));
                }
            }
            Action::RemoveSection(s) => self.ask_remove(Removal::Section(s)),
            Action::SetFolderEnabled(p, on) => self.set_folder_enabled(&p, on),
            Action::OpenPath(p) => self.open_path(&p),
            Action::Pick(p) => self.open_file_dialog(p),
            Action::Rescan => self.rescan(),
            Action::Undo => self.undo_removal(),
            Action::Show(p) => self.page = p,
        }
    }

    /// Open a folder or file with the desktop's handler.
    pub(crate) fn open_path(&mut self, p: &Path) {
        if let Err(e) = crate::net::open_external(&p.display().to_string()) {
            self.status = format!("Could not open {}: {e}", p.display());
        }
    }

    /// The issue report of `game`: its details, this machine, how its last
    /// run ended and the last lines of its log (the live log when that run
    /// is the latest, else its newest log file).
    pub(crate) fn issue_report(&self, game: &Game) -> crate::report::IssueReport {
        const LOG_LINES: usize = 80;
        let key = game.key();
        let last_run_here = matches!(&self.running, Some((_, RunKind::Game(k))) if *k == key);
        let log_tail = if last_run_here {
            let log = self.log.lock().unwrap_or_else(|e| e.into_inner());
            log.lines.iter().skip(log.lines.len().saturating_sub(LOG_LINES)).map(|l| l.text.clone()).collect()
        } else {
            launcher::latest_log_file(&launcher::logs_dir(), game.title_id.as_deref().unwrap_or(&game.title))
                .map(|p| crate::report::file_tail(&p, LOG_LINES))
                .unwrap_or_default()
        };
        let exit = self.crashed.as_ref().filter(|(k, _)| *k == key).map(|(_, e)| e.clone());
        crate::report::IssueReport {
            game_title: game.title.clone(),
            title_id: game.title_id.clone(),
            app_version: game.app_version.clone().or_else(|| game.version.clone()),
            emulator_version: env!("CARGO_PKG_VERSION").to_string(),
            os: crate::report::os_description(),
            gpu: self.gpu_name.clone(),
            exit,
            log_tail,
        }
    }

    fn report_issue(&mut self, game: &Game) {
        let url = self.issue_report(game).url();
        match crate::net::open_external(&url) {
            Ok(()) => self.status = "Opened the issue report in the browser".into(),
            Err(e) => self.status = format!("Could not open the browser: {e}"),
        }
    }

    /// After a game ended with an error: offers an issue report.
    fn crash_window(&mut self, ctx: &egui::Context) {
        if !self.crash_prompt {
            return;
        }
        let Some((key, how)) = self.crashed.clone() else { return };
        let Some(game) = self.game(&key).cloned() else {
            self.crash_prompt = false;
            return;
        };
        let mut open = true;
        let mut report = false;
        let mut close = false;
        let mut logs = false;
        egui::Window::new("The game stopped").open(&mut open).collapsible(false).resizable(false).show(ctx, |ui| {
            ui.label(format!("{} ended unexpectedly ({how}).", game.title));
            ui.label("You can report it: a prefilled issue opens in the browser with the game, your system and the last log lines.");
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                report = crate::theme::primary_button(ui, true, "Report issue", egui::vec2(120.0, 30.0)).clicked();
                logs = crate::theme::button(ui, true, "Show log", egui::vec2(100.0, 30.0)).clicked();
                close = crate::theme::button(ui, true, "Close", egui::vec2(80.0, 30.0)).clicked();
            });
        });
        if report {
            self.report_issue(&game);
        }
        if logs {
            self.page = Page::Logs;
            self.log_view.level = LevelFilter::All;
        }
        // The failure stays in later reports of the game (its menu).
        if report || close || logs || !open {
            self.crash_prompt = false;
        }
    }

    fn update_window(&mut self, ctx: &egui::Context) {
        let Some(r) = self.update_result.clone() else { return };
        let mut open = true;
        egui::Window::new("Update check").open(&mut open).collapsible(false).show(ctx, |ui| match &r {
            UpdateResult::UpToDate(tag) => {
                ui.label(format!("You are using the latest version ({tag})."));
            }
            UpdateResult::Available(info) => {
                ui.label(format!("An update is available.\n\nCurrent: {}\nLatest: {}", crate::updates::current_tag(), info.tag));
                if ui.button("Open the release page").clicked() {
                    let _ = crate::net::open_external(&info.page_url);
                }
            }
            UpdateResult::Failed(e) => {
                ui.label(format!("Could not check for updates:\n{e}"));
            }
        });
        if !open {
            self.update_result = None;
        }
    }

    /// A child process ended: remember how a game run ended, offer a report
    /// after a crash, and add an installed package's folder to the library.
    fn on_exit(&mut self, st: std::process::ExitStatus) {
        let Some((r, kind)) = self.running.as_ref() else { return };
        let (stopped, ran) = (r.stopped, r.ran_for.unwrap_or_default());
        match kind.clone() {
            RunKind::Game(k) => {
                let result = if stopped { "stopped".to_string() } else { launcher::describe_exit(&st) };
                let ok = stopped || st.success();
                if !ok {
                    self.crashed = Some((k.clone(), result.clone()));
                    self.crash_prompt = true;
                }
                self.status = format!("{} {}", r.title, if ok { format!("ended ({result})") } else { format!("failed: {result}") });
                self.settings.last_runs.insert(k, LastRun { ended: crate::theme::now_secs(), seconds: ran.as_secs(), result, ok });
                self.save_settings();
                self.reload_after_game();
            }
            RunKind::Install(title, games_dir) => {
                if st.success() {
                    self.status = format!("Installed {title}");
                    if self.settings.add_library_dir(games_dir) {
                        self.save_settings();
                    }
                    self.rescan();
                } else {
                    self.status = format!("Installing {title} failed: {} (see Logs)", launcher::describe_exit(&st));
                }
            }
            RunKind::Recompile(title) | RunKind::ShaderCache(title) => {
                self.status = format!("{title}: {} {}", kind.label().to_lowercase(), if st.success() { "done".to_string() } else { format!("failed: {}", launcher::describe_exit(&st)) });
            }
        }
    }
}

/// Library actions collected while drawing.
#[derive(Clone, Debug)]
pub(crate) enum Action {
    Select(String),
    Launch(String),
    /// Open "Run with..." for a game.
    RunWith(String),
    /// Copy the command line of a game's run.
    CopyCommand(String),
    /// Stop the running process.
    Stop,
    OpenFolder(String),
    Trophies(String),
    TrophyOverview,
    Cheats(String),
    RemoveSaveData(String),
    EditConfig(String),
    ClearConfig(String),
    InputProfile(String),
    SetStatus(String, GameStatus),
    SetComment(String, String),
    /// Ask to remove a game from the library.
    RemoveGame(String),
    /// Ask to remove a library folder or the added games.
    RemoveSection(Section),
    /// Scan a library folder or not (it stays listed).
    SetFolderEnabled(PathBuf, bool),
    OpenPath(PathBuf),
    /// Open a file dialog.
    Pick(FilePurpose),
    Rescan,
    /// Undo the last removal from the library.
    Undo,
    Install(String),
    Recompile(String),
    Precompile(String),
    /// Reset a game's compiled shaders.
    ClearShaderCache(String),
    /// Ask to reset the compiled shaders of every game.
    ClearAllShaderCaches,
    ExportShaderCache(String),
    ImportShaderCache(String),
    OpenLogs,
    /// Open a prefilled issue report for the game in the browser.
    ReportIssue(String),
    Show(Page),
}

/// The file name ending of an exported shader cache.
pub(crate) const SHADER_CACHE_SUFFIX: &str = ".shadercache";

impl App {
    /// One frame of the launcher window.
    pub(crate) fn frame(&mut self, ctx: &egui::Context) {
        self.apply_theme(ctx);
        self.poll_scan(ctx);
        self.poll_background();
        self.poll_banner(ctx);
        self.update_controllers();
        self.feed_capture(ctx);
        let mut ended = None;
        if let Some((r, _)) = self.running.as_mut() {
            let was_running = r.exit.is_none();
            match r.poll(&self.log) {
                None => ctx.request_repaint_after(Duration::from_millis(200)),
                Some(st) if was_running => ended = Some(st),
                Some(_) => {}
            }
        }
        if let Some(st) = ended {
            self.on_exit(st);
        }
        if !self.bindings.capturing() && ctx.input(|i| i.key_pressed(egui::Key::F5)) {
            self.rescan();
        }
        let mut actions = Vec::new();
        egui::SidePanel::left("nav")
            .exact_width(212.0)
            .resizable(false)
            .frame(egui::Frame::new().fill(crate::theme::pal_ctx(ctx).chrome).inner_margin(egui::Margin::symmetric(14, 16)))
            .show(ctx, |ui| self.sidebar(ui, &mut actions));
        egui::TopBottomPanel::bottom("status")
            .exact_height(30.0)
            .frame(egui::Frame::new().fill(crate::theme::pal_ctx(ctx).chrome).inner_margin(egui::Margin::symmetric(14, 4)))
            .show(ctx, |ui| self.status_bar(ui, &mut actions));
        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(crate::theme::pal_ctx(ctx).bg).inner_margin(egui::Margin::symmetric(22, 18)))
            .show(ctx, |ui| match self.page {
                Page::Library => self.library_page(ui, &mut actions),
                Page::Settings => self.settings_page(ui),
                Page::Logs => self.logs_page(ui, &mut actions),
                Page::About => self.about_page(ui, &mut actions),
            });
        for a in actions {
            self.do_action(a);
        }
        if let Some(t) = self.copy_text.take() {
            ctx.copy_text(t);
        }
        self.game_config_window(ctx);
        self.cheats_window(ctx);
        self.trophy_window(ctx);
        self.trophy_overview_window(ctx);
        self.run_dialog_window(ctx);
        self.update_window(ctx);
        self.crash_window(ctx);
        self.confirm_window(ctx);
        self.file_dialog_window(ctx);
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.frame(ctx);
    }

    /// Keys whose release went to another window (a game started with
    /// Enter, Alt+Tab) are released when the focus comes back.
    fn raw_input_hook(&mut self, ctx: &egui::Context, raw: &mut egui::RawInput) {
        let down = ctx.input(|i| i.keys_down.clone());
        crate::input_ui::release_stale_keys(&down, raw, false);
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        if let Some((r, _)) = self.running.as_mut() {
            r.stop(&self.log);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::library::GameSource;
    use crate::settings::{LibraryFolder, LibraryView};

    fn game(path: &str, id: Option<&str>, title: &str) -> Game {
        Game {
            source: GameSource::Folder(path.into()),
            dir: path.into(),
            eboot: Path::new(path).join("eboot.bin"),
            title: title.into(),
            title_id: id.map(str::to_string),
            version: Some("01.000".into()),
            ..Default::default()
        }
    }

    fn app() -> App {
        let mut s = GuiSettings::default();
        s.default_library = false;
        s.library_folders = vec![LibraryFolder::new("/srv/ps5games"), LibraryFolder::new("/mnt/usb/games")];
        s.extra_games = vec!["/home/u/homebrew".into()];
        let mut app = App::with(s, None, None, String::new(), GameConfigs::default(), CompatibilityDb::default());
        app.games = vec![
            game("/srv/ps5games/SILENT HILL", Some("PPSA10112"), "SILENT HILL: The Short Message"),
            game("/srv/ps5games/ASTRO BOT", Some("PPSA06400"), "ASTRO BOT"),
            game("/srv/ps5games/Demon's Souls", Some("PPSA01341"), "Demon's Souls"),
            game("/mnt/usb/games/T2D", Some("PPSA19393"), "Terminator 2D: NO FATE"),
            game("/home/u/homebrew", None, "homebrew"),
        ];
        app
    }

    #[test]
    fn remove_folder_from_the_list_asks_then_removes_and_undoes() {
        let mut app = app();
        let before = app.settings.clone();
        app.do_action(Action::RemoveSection(Section::Folder("/srv/ps5games".into())));
        let c = app.confirm.as_ref().expect("asks first");
        assert!(c.text.contains("its 3 games") && c.text.contains("not deleted"), "{}", c.text);
        assert_eq!(c.action.yes_label(), "Remove");
        assert_eq!(app.games.len(), 5, "nothing removed before the answer");
        let Some(Confirm { action: ConfirmAction::RemoveFromLibrary(r), .. }) = app.confirm.take() else { panic!() };
        app.remove_from_library(r);
        assert_eq!(app.games.iter().map(|g| g.title.as_str()).collect::<Vec<_>>(), ["Terminator 2D: NO FATE", "homebrew"]);
        assert_eq!(app.settings.library_roots(), [PathBuf::from("/mnt/usb/games")]);
        assert!(app.status.contains("3 games") && app.undo.is_some(), "{}", app.status);
        app.do_action(Action::Undo);
        assert_eq!(app.settings, before);
        assert!(app.undo.is_none());
    }

    #[test]
    fn remove_one_game_from_its_menu_asks_and_hides_it() {
        let mut app = app();
        app.selected = Some("/srv/ps5games/ASTRO BOT".into());
        app.do_action(Action::RemoveGame("/srv/ps5games/ASTRO BOT".into()));
        let Some(Confirm { text, action: ConfirmAction::RemoveFromLibrary(r), .. }) = app.confirm.take() else { panic!() };
        assert!(text.contains("\"ASTRO BOT\" (PPSA06400)"), "{text}");
        app.remove_from_library(r);
        assert_eq!(app.games.len(), 4);
        assert!(app.selected.is_none());
        assert_eq!(app.settings.hidden_games.len(), 1);
        app.undo_removal();
        assert!(app.settings.hidden_games.is_empty());
        // The added games as a whole.
        app.do_action(Action::RemoveSection(Section::Added));
        let Some(Confirm { text, action: ConfirmAction::RemoveFromLibrary(r), .. }) = app.confirm.take() else { panic!() };
        assert!(text.contains("1 game added one by one"), "{text}");
        app.remove_from_library(r);
        assert!(app.settings.extra_games.is_empty());
    }

    #[test]
    fn folder_switch_keeps_the_folder_and_counts_games() {
        let mut app = app();
        let counts = app.folder_counts();
        assert_eq!(counts.get(Path::new("/srv/ps5games")), Some(&3));
        assert_eq!(counts.get(Path::new("/mnt/usb/games")), Some(&1));
        app.do_action(Action::SetFolderEnabled("/mnt/usb/games".into(), false));
        assert_eq!(app.settings.library_roots(), [PathBuf::from("/srv/ps5games")]);
        assert_eq!(app.settings.library_folders.len(), 2, "still listed");
        assert!(app.status.contains("No longer scanning"), "{}", app.status);
        assert!(app.scan_rx.is_some(), "rescans");
        app.do_action(Action::SetFolderEnabled("/mnt/usb/games".into(), true));
        assert_eq!(app.settings.library_roots().len(), 2);
    }

    #[test]
    fn games_sort_by_the_chosen_column() {
        let mut app = app();
        app.sizes.insert("/srv/ps5games/ASTRO BOT".into(), 5);
        app.sizes.insert("/home/u/homebrew".into(), 1);
        app.settings.last_runs.insert("/mnt/usb/games/T2D".into(), LastRun { ended: 100, ..Default::default() });
        let titles = |app: &App| app.visible_games().into_iter().map(|g| g.title).collect::<Vec<_>>();
        assert_eq!(titles(&app)[0], "ASTRO BOT");
        app.sort = (Column::LastRun, true);
        assert_eq!(titles(&app)[0], "Terminator 2D: NO FATE", "the last played first");
        app.sort = (Column::Size, false);
        assert_eq!(titles(&app)[..2], ["ASTRO BOT", "homebrew"]);
        app.filter = "ppsa1".into();
        assert_eq!(titles(&app), ["Terminator 2D: NO FATE", "SILENT HILL: The Short Message"]);
    }

    #[test]
    fn run_with_options_reach_the_command_line() {
        let app = app();
        let g = app.games[1].clone();
        let plain = app.command_line(&g, None).unwrap();
        assert!(plain.contains("RUST_LOG=info") && plain.contains(" run ") && plain.contains("ASTRO BOT/eboot.bin"), "{plain}");
        assert!(!plain.contains("RPS5E_GPU_PROFILE"));
        let o = RunOptions { rust_log: "debug".into(), gpu_profile: 1, hle_trace: true, ..Default::default() };
        let line = app.command_line(&g, Some(&o)).unwrap();
        assert!(line.contains("RUST_LOG=debug") && !line.contains("RUST_LOG=info"), "{line}");
        assert!(line.contains("RPS5E_GPU_PROFILE=1") && line.contains("PS5_HLE_TRACE="), "{line}");
        let (_, env) = app.run_plan(&g, Some(&o)).unwrap();
        let trace = env.iter().find(|(k, _)| k == "PS5_HLE_TRACE").map(|(_, v)| PathBuf::from(v)).unwrap();
        assert!(trace.ends_with("debug/PPSA06400/hle.txt"), "{}", trace.display());
        let bad = RunOptions { compile_wait: "soon".into(), ..Default::default() };
        assert!(app.command_line(&g, Some(&bad)).is_err());
        // The dialog starts from the options last used for the game.
        let mut app = app;
        app.settings.run_options.insert(g.key(), o.clone());
        app.do_action(Action::RunWith(g.key()));
        assert_eq!(app.run_dialog.as_ref().map(|d| d.options.clone()), Some(o));
    }

    #[cfg(unix)]
    #[test]
    fn a_failed_run_is_remembered_and_offers_a_report() {
        let mut app = app();
        let key = app.games[0].key();
        let args = ["-c".into(), "echo boom >&2; exit 3".into()];
        let r = launcher::spawn(Path::new("sh"), &args, Path::new("/"), "SILENT HILL", &[], &app.log, None).unwrap();
        app.running = Some((r, RunKind::Game(key.clone())));
        let st = loop {
            if let Some(st) = app.running.as_mut().unwrap().0.poll(&app.log) {
                break st;
            }
            std::thread::sleep(Duration::from_millis(5));
        };
        assert!(app.running.as_ref().unwrap().0.command_line.contains("exit 3"));
        app.on_exit(st);
        let last = &app.settings.last_runs[&key];
        assert_eq!(last.result, "exit code 3");
        assert!(!last.ok && last.ended > 0);
        assert!(app.crash_prompt && app.crashed.as_ref().is_some_and(|(k, _)| *k == key));
        assert!(app.status.contains("failed"), "{}", app.status);
    }

    /// A frame of the launcher as a PNG file in `out` (10 frames, so
    /// layout and textures settle).
    fn shoot(app: &mut App, r: &mut crate::softrender::SoftRenderer, out: &Path, name: &str) {
        let mut img = (0, 0, Vec::new());
        for _ in 0..10 {
            img = r.frame(Vec::new(), |ctx| app.frame(ctx));
        }
        crate::softrender::write_png(&out.join(name), img.0, img.1, &img.2);
    }

    /// `RPS5E_GUI_RENDER_DIR=<dir> cargo test -p rps5e-gui render_library -- --ignored`
    /// writes the launcher's pages (library grid and list with details,
    /// settings, logs, about, Run with, the removal confirmation and Undo,
    /// the light theme) as PNG files.
    #[test]
    #[ignore]
    fn render_library() {
        let Some(out) = std::env::var_os("RPS5E_GUI_RENDER_DIR") else { return };
        let out = PathBuf::from(out);
        std::fs::create_dir_all(&out).unwrap();
        let mut app = app();
        app.settings.theme = Theme::Dark;
        app.settings.library_folders.push(LibraryFolder { path: "/media/old-disk".into(), enabled: false });
        app.compat.entries.insert("PPSA10112".into(), crate::compat::Entry { status: GameStatus::InGame, comment: "Plays to the end".into() });
        app.settings.last_runs.insert(
            "/srv/ps5games/SILENT HILL".into(),
            LastRun { ended: crate::theme::now_secs() - 3600, seconds: 1500, result: "ok".into(), ok: true },
        );
        for line in [
            "[2026-10-26T10:00:00Z INFO  ps5core] loading eboot.bin",
            "[2026-10-26T10:00:01Z WARN  ps5hle] sceKernelX: not implemented",
            "[2026-10-26T10:00:02Z ERROR ps5gpu] pipeline failed",
        ] {
            launcher::log_push(&app.log, launcher::classify(line), line);
        }
        let mut r = crate::softrender::SoftRenderer::new([1280.0, 800.0], 1.0);
        shoot(&mut app, &mut r, &out, "library-grid-no-selection.png");
        app.selected = Some("/srv/ps5games/SILENT HILL".into());
        shoot(&mut app, &mut r, &out, "library-grid.png");
        app.settings.view = LibraryView::List;
        shoot(&mut app, &mut r, &out, "library-list.png");
        app.do_action(Action::RunWith("/srv/ps5games/SILENT HILL".into()));
        app.run_dialog.as_mut().unwrap().options =
            RunOptions { gpu_profile: 1, hle_trace: true, display_resolution: "1080p".into(), ..Default::default() };
        shoot(&mut app, &mut r, &out, "run-with.png");
        app.run_dialog = None;
        for (tab, name) in [(Tab::Library, "settings-library.png"), (Tab::Emulator, "settings-emulator.png"), (Tab::Interface, "settings-interface.png")] {
            app.page = Page::Settings;
            app.tab = tab;
            shoot(&mut app, &mut r, &out, name);
        }
        app.page = Page::Logs;
        shoot(&mut app, &mut r, &out, "logs.png");
        app.page = Page::About;
        shoot(&mut app, &mut r, &out, "about.png");
        app.page = Page::Library;
        app.do_action(Action::RemoveSection(Section::Folder("/srv/ps5games".into())));
        shoot(&mut app, &mut r, &out, "library-remove-folder.png");
        let Some(Confirm { action: ConfirmAction::RemoveFromLibrary(rm), .. }) = app.confirm.take() else { panic!() };
        app.remove_from_library(rm);
        shoot(&mut app, &mut r, &out, "library-removed-undo.png");
        app.undo_removal();
        app.scan_rx = None;
        app.games = self::app().games;
        app.settings.theme = Theme::Light;
        app.selected = Some("/srv/ps5games/ASTRO BOT".into());
        app.settings.view = LibraryView::Grid;
        shoot(&mut app, &mut r, &out, "library-light.png");
    }

    /// `RPS5E_GUI_RENDER_DIR=<dir> GAMES_DIR=<games> cargo test -p rps5e-gui render_library_scan -- --ignored`
    /// scans a real games folder (`RPS5E_GUI_SCAN_DIR`, else `GAMES_DIR`:
    /// titles, covers, sizes) and writes the library as
    /// `library-games.png` (grid) and `library-games-list.png`.
    /// `RPS5E_GUI_RENDER_SIZE=<w>x<h>` sets the window size (points).
    #[test]
    #[ignore]
    fn render_library_scan() {
        let Some(out) = std::env::var_os("RPS5E_GUI_RENDER_DIR") else { return };
        let Some(games) = std::env::var_os("RPS5E_GUI_SCAN_DIR").or_else(|| std::env::var_os(settings::GAMES_DIR_ENV)) else { return };
        let out = PathBuf::from(out);
        std::fs::create_dir_all(&out).unwrap();
        let mut s = GuiSettings::default();
        s.default_library = false;
        s.theme = Theme::Dark;
        s.library_folders = vec![LibraryFolder::new(PathBuf::from(games))];
        let mut app = App::with(s, None, None, String::new(), GameConfigs::default(), CompatibilityDb::default());
        app.rescan();
        let size = std::env::var("RPS5E_GUI_RENDER_SIZE")
            .ok()
            .and_then(|v| v.split_once('x').and_then(|(w, h)| Some([w.parse().ok()?, h.parse().ok()?])));
        let mut r = crate::softrender::SoftRenderer::new(size.unwrap_or([1280.0, 800.0]), 1.0);
        let start = std::time::Instant::now();
        while app.scan_rx.is_some() && start.elapsed() < Duration::from_secs(120) {
            r.frame(Vec::new(), |ctx| app.frame(ctx));
            std::thread::sleep(Duration::from_millis(20));
        }
        app.selected = app.visible_games().first().map(|g| g.key());
        // The selected game's background picture loads in the background.
        let start = std::time::Instant::now();
        while app.selected.as_ref().is_some_and(|k| !app.banners.contains_key(k)) && start.elapsed() < Duration::from_secs(30) {
            r.frame(Vec::new(), |ctx| app.frame(ctx));
            std::thread::sleep(Duration::from_millis(20));
        }
        shoot(&mut app, &mut r, &out, "library-games.png");
        app.settings.view = LibraryView::List;
        shoot(&mut app, &mut r, &out, "library-games-list.png");
    }
}
