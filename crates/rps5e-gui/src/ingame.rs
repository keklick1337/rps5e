// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The in-game menu: opened over the running game with F1 (action
//! `ToggleMenu`) or the controller's PS button, drawn with egui by the
//! game window (`ps5gpu::menu`). Sections:
//!
//! - **Controls**: device mode, controller per player, controller features,
//!   sticks and mouse, the bindings on the DualSense picture or as a table
//!   (the launcher's widgets, [`crate::input_ui`], [`crate::dualsense`]);
//!   saved to the game's input profile and applied to the running game at
//!   once ([`ps5input::reconfigure`]).
//! - **Controls hint**: the same picture read-only over the game, lighting
//!   up the buttons whose inputs are held; Esc, the menu key, Options or
//!   PS close it. Its full version, with every binding listed, shows while
//!   the controls key (F2) is held, the menu closed and the game running
//!   ([`crate::controls_overlay`], drawn by [`InGameMenu`]'s overlay).
//! - **Graphics**: present mode, fullscreen, scaling and its filter, the
//!   performance overlay, the FPS counter and the per-draw GPU fast paths
//!   apply at once; HDR output, the guest refresh rate and the GPU switches
//!   that are read at start ([`GPU_SWITCHES`]) are marked "after restart".
//! - **Audio**: master volume.
//! - **System**: console language (after restart), pausing the game while
//!   the menu is open, exit to the launcher.
//!
//! Settings are saved to the game's configuration in `games.toml` (the
//! launcher's per-game configuration, created from the global settings
//! when the game has none) and reach the next start as environment
//! variables. While the menu is open the game gets no input (neutral
//! pads); with "pause" on, its threads also stop at the next frame
//! ([`ps5hle::pause`]).

use crate::config::{self, EmulatorSettings, GameConfig, GameConfigs};
use crate::input_store::{InputStore, Target};
use crate::dualsense;
use crate::input_ui::{self, BindingEditor, BindingSet};
use crate::settings::GuiSettings;
use eframe::egui::{self, Color32, RichText};
use ps5gpu::display::{self, PresentMode, ScaleFilter, ScaleMode};
use ps5input::mapping::{Config, Control, DeviceMode, Source};
use ps5input::menu::Nav;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

/// Set by the launcher for the games it starts: the game's key in
/// `games.toml`, its title, and that the launcher is still open.
pub const GAME_KEY_ENV: &str = "RPS5E_GAME_KEY";
pub const GAME_TITLE_ENV: &str = "RPS5E_GAME_TITLE";
pub const FROM_LAUNCHER_ENV: &str = "RPS5E_FROM_LAUNCHER";
/// `RPS5E_MENU_PAUSE=1`: the game pauses while the menu is open.
pub const MENU_PAUSE_ENV: &str = "RPS5E_MENU_PAUSE";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Section {
    Controls,
    Graphics,
    Audio,
    System,
}

impl Section {
    pub const ALL: [Section; 4] = [Section::Controls, Section::Graphics, Section::Audio, Section::System];

    pub fn label(self) -> &'static str {
        match self {
            Section::Controls => "Controls",
            Section::Graphics => "Graphics",
            Section::Audio => "Audio",
            Section::System => "System",
        }
    }

    fn index(self) -> usize {
        Self::ALL.iter().position(|s| *s == self).unwrap_or(0)
    }

    pub fn next(self) -> Self {
        Self::ALL[(self.index() + 1) % Self::ALL.len()]
    }

    pub fn prev(self) -> Self {
        Self::ALL[(self.index() + Self::ALL.len() - 1) % Self::ALL.len()]
    }
}

/// A GPU switch read once when the game starts (changes apply after a
/// restart; kept in the game configuration's environment list).
#[derive(Clone, Copy, Debug)]
pub struct GpuSwitch {
    pub env: &'static str,
    pub label: &'static str,
    /// On unless set to `0`; otherwise off unless set to `1`.
    pub default_on: bool,
}

pub const GPU_SWITCHES: &[GpuSwitch] = &[
    GpuSwitch { env: "RPS5E_SAFE", label: "Safe mode: every GPU fast path off (diagnostic, slow)", default_on: false },
    GpuSwitch { env: "RPS5E_ASYNC_PIPELINES", label: "Compile pipelines in the background (draws wait less, may be skipped)", default_on: true },
    GpuSwitch { env: "RPS5E_SHADER_CACHE", label: "Shader disk cache", default_on: true },
    GpuSwitch { env: "RPS5E_BUFFER_MEMO", label: "Buffer memo", default_on: true },
    GpuSwitch { env: "RPS5E_VIEW_MEMO", label: "Texture view memo", default_on: true },
    GpuSwitch { env: "RPS5E_BIND_MEMO", label: "Binding memo", default_on: true },
    GpuSwitch { env: "RPS5E_STATE_FILTER", label: "Draw state filter", default_on: true },
    GpuSwitch { env: "RPS5E_SHARED_SCOPES", label: "Shared rendering scopes", default_on: true },
    GpuSwitch { env: "RPS5E_EXEC_RUN", label: "Shader EXEC runs", default_on: true },
    GpuSwitch { env: "RPS5E_GPU_INDIRECT", label: "Indirect draws read on the GPU", default_on: true },
    GpuSwitch { env: "RPS5E_ALIAS_MERGE", label: "Image alias merging", default_on: true },
    GpuSwitch { env: "RPS5E_GPU_ZERO_COPY", label: "Zero-copy GPU downloads", default_on: true },
    GpuSwitch { env: "RPS5E_BUFFER_SUBALLOC", label: "Buffer sub-allocation", default_on: true },
    GpuSwitch { env: "RPS5E_BDA", label: "Shaders read guest memory directly (BDA, experimental)", default_on: false },
    GpuSwitch { env: "RPS5E_GPU_RECORD_THREAD", label: "Separate recording thread (while recording is the bottleneck)", default_on: true },
    GpuSwitch { env: "RPS5E_OCCLUSION_QUERIES", label: "Occlusion queries on the GPU", default_on: false },
];

impl GpuSwitch {
    /// Whether the switch is on for a variable value (`None`: unset).
    pub fn is_on(&self, value: Option<&str>) -> bool {
        let v = value.map(str::trim);
        if self.default_on {
            !v.is_some_and(|v| v == "0" || v.eq_ignore_ascii_case("off"))
        } else {
            v == Some("1")
        }
    }

    /// The value in a `KEY=VALUE` list (the last one wins, like the
    /// environment the launcher builds).
    fn value_in<'a>(&self, env: &'a [String]) -> Option<&'a str> {
        env.iter().rev().find_map(|e| e.split_once('=').filter(|(k, _)| k.trim() == self.env).map(|(_, v)| v))
    }

    pub fn on_in(&self, env: &[String]) -> bool {
        self.is_on(self.value_in(env))
    }

    /// Set the switch in a `KEY=VALUE` list (nothing for the default).
    pub fn set_in(&self, env: &mut Vec<String>, on: bool) {
        env.retain(|e| e.split_once('=').is_none_or(|(k, _)| k.trim() != self.env));
        if on != self.default_on {
            env.push(format!("{}={}", self.env, if on { "1" } else { "0" }));
        }
    }
}

/// The settings this run started with that only apply at start.
#[derive(Clone, Debug, PartialEq)]
pub struct Started {
    pub console_language: i32,
    pub refresh_rate: f64,
    /// HDR output (`RPS5E_HDR`).
    pub hdr: bool,
    /// Emulated display resolution (`RPS5E_DISPLAY_RESOLUTION`), as the
    /// launcher stores it (`720p`, `1080p`, `2160p`).
    pub display_resolution: String,
    /// [`GPU_SWITCHES`] in order.
    pub switches: Vec<bool>,
}

impl Started {
    /// From a variable lookup (the process environment).
    pub fn from_vars(var: impl Fn(&str) -> Option<String>) -> Self {
        Started {
            console_language: var("RPS5E_SYSTEM_LANGUAGE")
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(config::DEFAULT_CONSOLE_LANGUAGE),
            refresh_rate: var("RPS5E_GUEST_REFRESH_RATE").and_then(|v| v.trim().parse().ok()).unwrap_or(0.0),
            hdr: var("RPS5E_HDR").is_some_and(|v| ps5gpu::display::parse_hdr(&v)),
            display_resolution: var("RPS5E_DISPLAY_RESOLUTION")
                .and_then(|v| ps5gpu::display::DisplayResolution::parse(&v))
                .unwrap_or_default()
                .as_str()
                .into(),
            switches: GPU_SWITCHES.iter().map(|s| s.is_on(var(s.env).as_deref())).collect(),
        }
    }

    pub fn from_env() -> Self {
        Self::from_vars(|k| std::env::var(k).ok())
    }
}

/// Settings of `cfg` that differ from what the game started with.
pub fn restart_pending(cfg: &EmulatorSettings, started: &Started) -> Vec<&'static str> {
    let mut v = Vec::new();
    if cfg.console_language != started.console_language {
        v.push("console language");
    }
    if cfg.refresh_rate != started.refresh_rate {
        v.push("refresh rate");
    }
    if cfg.hdr != started.hdr {
        v.push("HDR");
    }
    if cfg.display_resolution != started.display_resolution {
        v.push("display resolution");
    }
    for (sw, &was) in GPU_SWITCHES.iter().zip(&started.switches) {
        if sw.on_in(&cfg.extra_env) != was {
            v.push(sw.env);
        }
    }
    v
}

/// A setting that applies to the running game at once.
#[derive(Clone, Debug, PartialEq)]
pub enum LiveChange {
    PresentMode(PresentMode),
    Fullscreen(bool),
    PerfOverlay(bool),
    FpsCounter(bool),
    ScaleMode(ScaleMode),
    ScaleFilter(ScaleFilter),
    FastPaths(bool),
    Volume(u32),
    PauseInMenu(bool),
}

/// The live changes from `before` to `after`.
pub fn live_changes(before: &EmulatorSettings, after: &EmulatorSettings) -> Vec<LiveChange> {
    let mut v = Vec::new();
    if before.present_mode != after.present_mode {
        if let Some(m) = PresentMode::parse(&after.present_mode) {
            v.push(LiveChange::PresentMode(m));
        }
    }
    if before.fullscreen != after.fullscreen {
        v.push(LiveChange::Fullscreen(after.fullscreen));
    }
    if before.perf_overlay != after.perf_overlay {
        v.push(LiveChange::PerfOverlay(after.perf_overlay));
    }
    if before.fps_counter != after.fps_counter {
        v.push(LiveChange::FpsCounter(after.fps_counter));
    }
    if before.scale_mode != after.scale_mode {
        if let Some(m) = ScaleMode::parse(&after.scale_mode) {
            v.push(LiveChange::ScaleMode(m));
        }
    }
    if before.scale_filter != after.scale_filter {
        if let Some(f) = ScaleFilter::parse(&after.scale_filter) {
            v.push(LiveChange::ScaleFilter(f));
        }
    }
    if before.fast_paths != after.fast_paths {
        v.push(LiveChange::FastPaths(after.fast_paths));
    }
    if before.volume != after.volume {
        v.push(LiveChange::Volume(after.volume.min(100)));
    }
    if before.pause_in_menu != after.pause_in_menu {
        v.push(LiveChange::PauseInMenu(after.pause_in_menu));
    }
    v
}

/// Pause the game while the menu is open (initially `RPS5E_MENU_PAUSE`).
static PAUSE_IN_MENU: AtomicBool = AtomicBool::new(false);

pub fn apply_live(c: &LiveChange) {
    match *c {
        LiveChange::PresentMode(m) => display::set_present_mode(m),
        LiveChange::Fullscreen(on) => ps5gpu::input::request_window(ps5gpu::input::WindowRequest::SetFullscreen(on)),
        LiveChange::PerfOverlay(on) => ps5gpu::perf::set_overlay_enabled(on),
        LiveChange::FpsCounter(on) => display::set_fps_counter(on),
        LiveChange::ScaleMode(m) => display::set_scale_mode(m),
        LiveChange::ScaleFilter(f) => display::set_scale_filter(f),
        LiveChange::FastPaths(on) => ps5gpu::backend::ab::set_live_fast(on),
        LiveChange::Volume(v) => ps5hle::audio::host::set_master_volume(v as f32 / 100.0),
        LiveChange::PauseInMenu(on) => {
            PAUSE_IN_MENU.store(on, Ordering::Relaxed);
            ps5hle::pause::set_paused(on && ps5gpu::menu::is_open());
        }
    }
}

/// `base` with the live settings as the running game has them now.
pub fn current_live(base: &EmulatorSettings) -> EmulatorSettings {
    let mut s = base.clone();
    s.present_mode = display::present_mode().as_str().into();
    s.fullscreen = ps5gpu::input::fullscreen();
    s.perf_overlay = ps5gpu::perf::overlay_enabled();
    s.fps_counter = display::fps_counter();
    s.scale_mode = display::scale_mode().as_str().into();
    s.scale_filter = display::scale_filter().as_str().into();
    s.fast_paths = ps5gpu::backend::ab::live_fast();
    s.volume = (ps5hle::audio::host::master_volume() * 100.0).round() as u32;
    s.pause_in_menu = PAUSE_IN_MENU.load(Ordering::Relaxed);
    s
}

/// The game this process runs.
#[derive(Clone, Debug, PartialEq)]
pub struct GameInfo {
    /// Key of its configuration in `games.toml` (the game folder or package).
    pub key: String,
    pub title: String,
    pub title_id: Option<String>,
}

impl GameInfo {
    /// For the eboot being run (the launcher names key and title).
    pub fn detect(eboot: &Path) -> Self {
        let title_id = ps5hle::app_info::title_id().filter(|t| ps5input::profile::valid_title_id(t));
        let dir = eboot.parent().map(|d| std::fs::canonicalize(d).unwrap_or_else(|_| d.to_path_buf()));
        let key = std::env::var(GAME_KEY_ENV)
            .ok()
            .filter(|k| !k.is_empty())
            .or_else(|| dir.as_ref().map(|d| d.display().to_string()))
            .unwrap_or_default();
        let title = std::env::var(GAME_TITLE_ENV)
            .ok()
            .filter(|t| !t.is_empty())
            .or_else(|| title_id.clone())
            .or_else(|| dir.as_ref().and_then(|d| d.file_name()).map(|n| n.to_string_lossy().into_owned()))
            .unwrap_or_else(|| "game".into());
        GameInfo { key, title, title_id }
    }
}

/// The folder of the input settings the running game reads.
fn input_dir(config_dir: Option<&Path>) -> Option<PathBuf> {
    match ps5input::mapping::global_config_path() {
        Some(p) if p.file_name().is_some_and(|n| n == ps5input::mapping::FILE_NAME) => {
            p.parent().filter(|d| !d.as_os_str().is_empty()).map(Path::to_path_buf).or_else(|| Some(PathBuf::from(".")))
        }
        _ => config_dir.map(Path::to_path_buf),
    }
}

/// The menu's state (everything but drawing; no window needed).
pub struct MenuState {
    pub section: Section,
    pub game: GameInfo,
    pub config_dir: Option<PathBuf>,
    /// Folder of the input settings (global file and profiles/).
    pub input_dir: Option<PathBuf>,
    /// The game's settings (its configuration, else the global ones) with
    /// the live values of the running game.
    pub settings: EmulatorSettings,
    /// The game has its own configuration in `games.toml`.
    pub own_config: bool,
    pub started: Started,
    pub input: Option<InputStore>,
    /// The input settings when the menu opened (Undo).
    pub input_at_open: Option<Config>,
    pub bindings: BindingEditor,
    pub status: Option<String>,
    pub error: Option<String>,
    pub from_launcher: bool,
    pub confirm_exit: bool,
    /// The read-only controls hint is shown instead of the menu.
    pub hint: bool,
    dirty: bool,
    input_dirty: bool,
    focus_pending: bool,
    focus_confirm: bool,
    close_request: bool,
    /// "Recompile shaders" was chosen: the menu closes and the rebuild starts.
    recompile_request: bool,
}

impl MenuState {
    pub fn new(game: GameInfo, config_dir: Option<PathBuf>, input_dir: Option<PathBuf>, started: Started) -> Self {
        let (settings, own_config, error) = match config_dir.as_deref() {
            Some(d) => match Self::load_settings(d, &game.key) {
                Ok((s, own)) => (s, own, None),
                Err(e) => (EmulatorSettings::default(), false, Some(e)),
            },
            None => (EmulatorSettings::default(), false, Some("no configuration folder: settings are not saved".into())),
        };
        let mut st = MenuState {
            section: Section::Controls,
            game,
            config_dir,
            input_dir,
            settings,
            own_config,
            started,
            input: None,
            input_at_open: None,
            bindings: BindingEditor::default(),
            status: None,
            error,
            from_launcher: std::env::var_os(FROM_LAUNCHER_ENV).is_some(),
            confirm_exit: false,
            hint: false,
            dirty: false,
            input_dirty: false,
            focus_pending: true,
            focus_confirm: false,
            close_request: false,
            recompile_request: false,
        };
        st.load_input();
        st
    }

    /// The game's configuration in `dir`, else the global settings;
    /// whether the game has its own.
    pub fn load_settings(dir: &Path, key: &str) -> Result<(EmulatorSettings, bool), String> {
        let configs = GameConfigs::load(dir)?;
        if let Some(g) = configs.games.get(key) {
            return Ok((g.settings.clone(), true));
        }
        let global = GuiSettings::load_from(&dir.join(crate::settings::FILE_NAME))?;
        Ok((global.emulator, false))
    }

    fn load_input(&mut self) {
        let Some(dir) = self.input_dir.clone() else { return };
        let target = match &self.game.title_id {
            Some(id) => Target::Game { title_id: id.clone(), title: self.game.title.clone() },
            None => Target::Global,
        };
        match InputStore::load(&dir, target) {
            Ok(s) => {
                self.input_at_open = Some(s.config.clone());
                self.input = Some(s);
            }
            Err(e) => self.error = Some(e),
        }
    }

    /// The menu opened: show the running game's live values, reload the
    /// input settings (they may have been edited elsewhere).
    pub fn on_open(&mut self) {
        self.settings = current_live(&self.settings);
        self.confirm_exit = false;
        self.hint = false;
        self.focus_pending = true;
        self.bindings.capture = None;
        if !self.input_dirty {
            self.load_input();
        }
        if let Some(r) = ps5gpu::pipeline::precompile::take_recompile_result() {
            self.status = Some(r);
        }
    }

    /// Change the settings: live ones apply now; all are saved.
    pub fn set_settings(&mut self, next: EmulatorSettings) {
        if next == self.settings {
            return;
        }
        for c in live_changes(&self.settings, &next) {
            apply_live(&c);
        }
        self.settings = next;
        self.dirty = true;
    }

    /// Write the settings into the game's configuration.
    pub fn save(&mut self) -> Result<PathBuf, String> {
        let dir = self.config_dir.clone().ok_or("no configuration folder")?;
        let mut configs = GameConfigs::load(&dir)?;
        let entry = configs.games.entry(self.game.key.clone()).or_insert_with(|| GameConfig {
            title_id: self.game.title_id.clone().unwrap_or_default(),
            name: self.game.title.clone(),
            settings: EmulatorSettings::default(),
        });
        entry.settings = self.settings.clone();
        configs.save(&dir)?;
        self.own_config = true;
        self.dirty = false;
        Ok(GameConfigs::path(&dir))
    }

    /// Change the input settings: applied to the running game now.
    pub fn set_input(&mut self, cfg: Config) {
        let Some(store) = self.input.as_mut() else { return };
        if store.config == cfg {
            return;
        }
        store.config = cfg;
        store.dirty = true;
        self.input_dirty = true;
        ps5input::reconfigure(&store.config);
    }

    /// Turn the game's input profile on (it starts from the settings in
    /// effect) or off (the global settings apply again).
    pub fn set_profile_enabled(&mut self, on: bool) {
        let Some(store) = self.input.as_mut() else { return };
        store.set_profile_enabled(on);
        self.input_dirty = true;
        let effective = if on {
            Some(store.config.clone())
        } else {
            InputStore::load(&store.dir, Target::Global).ok().map(|g| g.config)
        };
        if let Some(cfg) = effective {
            ps5input::reconfigure(&cfg);
        }
    }

    pub fn save_input(&mut self) -> Result<Option<PathBuf>, String> {
        let Some(store) = self.input.as_mut() else { return Ok(None) };
        store.save()?;
        self.input_dirty = false;
        Ok(Some(store.path()))
    }

    /// Save what changed (called when no drag is in progress).
    pub fn flush(&mut self) {
        if self.dirty {
            match self.save() {
                Ok(p) => self.status = Some(format!("Saved to the game configuration ({})", p.display())),
                Err(e) => self.error = Some(format!("cannot save the settings: {e}")),
            }
        }
        if self.input_dirty {
            match self.save_input() {
                Ok(Some(p)) => self.status = Some(format!("Saved the input settings ({})", p.display())),
                Ok(None) => {}
                Err(e) => self.error = Some(format!("cannot save the input settings: {e}")),
            }
        }
    }

    /// Show the controls hint, for the set the device mode uses.
    pub fn open_hint(&mut self) {
        self.hint = true;
        self.confirm_exit = false;
        self.bindings.capture = None;
        let mode = self.input.as_ref().map(|s| s.config.mode).unwrap_or_default();
        self.bindings.set = hint_set(mode);
    }

    /// The input settings the running game uses (the store's when host
    /// input is not running).
    pub fn effective_input(&self) -> Config {
        ps5input::active_config().or_else(|| self.input.as_ref().map(|s| s.config.clone())).unwrap_or_default()
    }

    /// Where the bindings in effect come from.
    pub fn bindings_source(&self) -> String {
        match self.input.as_ref().map(|s| (&s.target, s.profile_enabled)) {
            Some((Target::Game { title_id, .. }, true)) => format!("Per-game profile ({title_id})"),
            _ => "Global settings".into(),
        }
    }

    /// Keys bound to the menu (it closes with them too).
    pub fn menu_keys(&self) -> Vec<ps5gpu::input::KeyCode> {
        let cfg = self.input.as_ref().map(|s| s.config.clone()).unwrap_or_default();
        cfg.sources_of(Control::ToggleMenu, false)
            .into_iter()
            .chain(cfg.sources_of(Control::ToggleMenu, true))
            .filter_map(|s| match s {
                Source::Key(k) => Some(k),
                _ => None,
            })
            .collect()
    }
}

/// Start the launcher (when the game was not started from one) and end
/// the emulator.
pub fn exit_to_launcher(from_launcher: bool) {
    if !from_launcher {
        spawn_launcher();
    }
    ps5hle::pause::set_paused(false);
    ps5vm::shutdown::request(0);
}

/// This program without arguments (the launcher), without the game's run
/// settings in its environment.
fn spawn_launcher() {
    let Ok(exe) = std::env::current_exe() else { return };
    let mut cmd = std::process::Command::new(exe);
    for (k, _) in std::env::vars_os() {
        let k = k.to_string_lossy().into_owned();
        if (k.starts_with("RPS5E_") && k != "RPS5E_CONFIG_DIR") || k == "RUST_LOG" {
            cmd.env_remove(k);
        }
    }
    cmd.stdin(std::process::Stdio::null()).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    if let Err(e) = cmd.spawn() {
        log::warn!("cannot start the launcher: {e}");
    }
}

/// Navigation steps as egui key presses (directions, Enter, Escape).
fn nav_key(n: Nav) -> Option<egui::Key> {
    Some(match n {
        Nav::Up => egui::Key::ArrowUp,
        Nav::Down => egui::Key::ArrowDown,
        Nav::Left => egui::Key::ArrowLeft,
        Nav::Right => egui::Key::ArrowRight,
        Nav::Accept => egui::Key::Enter,
        _ => return None,
    })
}

fn key_event(key: egui::Key, pressed: bool) -> egui::Event {
    egui::Event::Key { key, physical_key: None, pressed, repeat: false, modifiers: egui::Modifiers::NONE }
}

/// What a frame's controller steps do besides key presses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct NavActions {
    close: bool,
    back: bool,
    tab: i32,
}

/// Set when the menu opens (by any input), taken by its next UI frame.
static JUST_OPENED: AtomicBool = AtomicBool::new(false);

/// The menu installed into the game window.
pub struct InGameMenu {
    pub state: MenuState,
    actions: NavActions,
    /// Keys egui held at the end of the last frame.
    keys_down: std::collections::HashSet<egui::Key>,
}

impl InGameMenu {
    pub fn new(state: MenuState) -> Self {
        InGameMenu { state, actions: NavActions::default(), keys_down: Default::default() }
    }

    /// The menu opened. Keys pressed while it was open last time were
    /// released to the game: egui gets their releases now.
    fn opened(&mut self, raw: &mut egui::RawInput) {
        input_ui::release_stale_keys(&self.keys_down, raw, true);
        self.state.on_open();
        self.actions = NavActions::default();
    }

    fn close(&mut self) {
        self.state.flush();
        self.state.bindings.capture = None;
        self.state.hint = false;
        ps5gpu::menu::set_open(false);
    }
}

impl ps5gpu::menu::MenuApp for InGameMenu {
    fn raw_input(&mut self, raw: &mut egui::RawInput) {
        let steps = ps5input::menu::take_nav();
        if JUST_OPENED.swap(false, Ordering::AcqRel) {
            self.opened(raw);
            return;
        }
        input_ui::release_stale_keys(&self.keys_down, raw, false);
        // While a binding is captured the controller is the input; the
        // hint only closes (its buttons light up instead of navigating).
        if self.state.bindings.capturing() {
            return;
        }
        if self.state.hint {
            self.actions.close |= steps.contains(&Nav::Close);
            return;
        }
        for n in steps {
            match n {
                Nav::Close => self.actions.close = true,
                Nav::Back => self.actions.back = true,
                Nav::PrevTab => self.actions.tab -= 1,
                Nav::NextTab => self.actions.tab += 1,
                other => {
                    if let Some(k) = nav_key(other) {
                        raw.events.push(key_event(k, true));
                        raw.events.push(key_event(k, false));
                        self.state.focus_pending |= matches!(other, Nav::Up | Nav::Down | Nav::Left | Nav::Right);
                    }
                }
            }
        }
    }

    fn overlay(&mut self, ctx: &egui::Context) {
        let cfg = self.state.effective_input();
        let source = self.state.bindings_source();
        crate::controls_overlay::draw(ctx, &crate::controls_overlay::OverlayView { cfg: &cfg, title: &self.state.game.title, source: &source });
    }

    fn ui(&mut self, ctx: &egui::Context) {
        self.keys_down = ctx.input(|i| i.keys_down.clone());
        let actions = std::mem::take(&mut self.actions);
        let popup = ctx.memory(|m| m.any_popup_open());
        let capturing = self.state.bindings.capturing();
        // Close: Resume, Escape / the menu key (not while binding), the
        // controller's Options, or Circle with nothing to back out of.
        let keys = self.state.menu_keys();
        let (escape, menu_key) = ctx.input(|i| {
            let esc = i.key_pressed(egui::Key::Escape);
            let menu = i.events.iter().any(|e| match e {
                egui::Event::Key { physical_key: Some(k), pressed: true, repeat: false, .. } => {
                    crate::keys::key_code(*k).is_some_and(|c| keys.contains(&c))
                }
                _ => false,
            });
            (esc, menu)
        });
        let mut close = !capturing && (actions.close || menu_key || (escape && !popup));
        if self.state.hint {
            if !close {
                draw_hint(ctx, &mut self.state);
                close = self.state.close_requested();
            }
            if close {
                self.close();
            }
            return;
        }
        if actions.back && !capturing {
            if popup {
                ctx.memory_mut(|m| m.close_popup());
            } else if self.state.confirm_exit {
                self.state.confirm_exit = false;
            } else {
                close = true;
            }
        }
        if actions.tab != 0 {
            let mut s = self.state.section;
            for _ in 0..actions.tab.unsigned_abs() {
                s = if actions.tab > 0 { s.next() } else { s.prev() };
            }
            self.state.section = s;
            self.state.focus_pending = true;
        }
        if close {
            self.close();
            return;
        }
        draw(ctx, &mut self.state);
        if self.state.close_requested() {
            let recompile = std::mem::take(&mut self.state.recompile_request);
            self.close();
            if recompile {
                recompile_shaders(&mut self.state);
            }
            return;
        }
        if !ctx.input(|i| i.pointer.any_down()) {
            self.state.flush();
        }
    }
}

impl MenuState {
    fn close_requested(&mut self) -> bool {
        std::mem::take(&mut self.close_request)
    }
}

/// Clear this game's shader and pipeline caches and build them again on
/// every core behind the compile overlay, the game paused meanwhile
///. The menu is closed first (closing resumes the game).
fn recompile_shaders(st: &mut MenuState) {
    ps5hle::pause::set_paused(true);
    let started = ps5gpu::pipeline::precompile::start_recompile(|outcome| {
        ps5hle::pause::set_paused(false);
        ps5gpu::osd::show_hint(outcome.to_string(), std::time::Duration::from_secs(8));
    });
    if !started {
        ps5hle::pause::set_paused(false);
        st.error = Some("Shaders cannot be recompiled now (no game title, no GPU, or a rebuild runs already).".into());
    }
}

/// Install the menu for the game `eboot` (before input starts).
pub fn install(eboot: &Path) {
    if std::env::var(MENU_PAUSE_ENV).is_ok_and(|v| v == "1") {
        PAUSE_IN_MENU.store(true, Ordering::Relaxed);
    }
    let config_dir = crate::settings::config_dir();
    let inputs = input_dir(config_dir.as_deref());
    let state = MenuState::new(GameInfo::detect(eboot), config_dir, inputs, Started::from_env());
    ps5gpu::menu::install(Box::new(InGameMenu::new(state)));
    ps5gpu::menu::on_open_change(|open| {
        if open {
            JUST_OPENED.store(true, Ordering::Release);
        }
        ps5hle::pause::set_paused(open && PAUSE_IN_MENU.load(Ordering::Relaxed));
    });
}

// ---------------------------------------------------------------------------
// Drawing
// ---------------------------------------------------------------------------

fn draw(ctx: &egui::Context, st: &mut MenuState) {
    // Dim the game behind the menu.
    let screen = ctx.screen_rect();
    egui::Area::new(egui::Id::new("menu-backdrop")).order(egui::Order::Background).fixed_pos(screen.min).show(ctx, |ui| {
        ui.painter().rect_filled(screen, 0.0, Color32::from_black_alpha(150));
        ui.allocate_rect(screen, egui::Sense::click());
    });
    let size = egui::vec2((screen.width() - 32.0).clamp(320.0, 1040.0), (screen.height() - 32.0).clamp(240.0, 620.0));
    egui::Window::new(RichText::new(format!("{} - {}", ps5gpu::APP_TITLE, st.game.title)).strong())
        .id(egui::Id::new("ingame-menu"))
        .collapsible(false)
        .resizable(false)
        .title_bar(true)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .fixed_size(size)
        .frame(egui::Frame::window(&ctx.style()).fill(Color32::from_rgba_unmultiplied(24, 26, 30, 245)))
        .show(ctx, |ui| {
            ui.horizontal_top(|ui| {
                ui.vertical(|ui| {
                    ui.set_width(150.0);
                    let mut first = None;
                    for s in Section::ALL {
                        let r = ui.add_sized([140.0, 32.0], egui::SelectableLabel::new(st.section == s, RichText::new(s.label()).size(16.0)));
                        if r.clicked() && st.section != s {
                            st.section = s;
                            st.bindings.capture = None;
                        }
                        if s == st.section {
                            first = Some(r.id);
                        }
                    }
                    if std::mem::take(&mut st.focus_pending) && ui.ctx().memory(|m| m.focused().is_none()) {
                        if let Some(id) = first {
                            ui.ctx().memory_mut(|m| m.request_focus(id));
                        }
                    }
                    ui.add_space(12.0);
                    if ui.add_sized([140.0, 30.0], egui::Button::new("Resume")).clicked() {
                        st.close_request = true;
                    }
                    if ui
                        .add_sized([140.0, 30.0], egui::Button::new("Controls hint"))
                        .on_hover_text("The controller picture with the current bindings")
                        .clicked()
                    {
                        st.open_hint();
                    }
                    if ui.add_sized([140.0, 30.0], egui::Button::new("Exit to launcher")).clicked() {
                        st.confirm_exit = true;
                        st.focus_confirm = true;
                    }
                });
                ui.separator();
                ui.vertical(|ui| {
                    let pending = restart_pending(&st.settings, &st.started);
                    if !pending.is_empty() {
                        ui.colored_label(Color32::YELLOW, format!("Restart the game to apply: {}", pending.join(", ")));
                    }
                    if st.confirm_exit {
                        confirm_exit(ui, st);
                    }
                    egui::ScrollArea::vertical().auto_shrink([false, false]).max_height(ui.available_height() - 44.0).show(ui, |ui| match st.section {
                        Section::Controls => controls(ui, st),
                        Section::Graphics => graphics(ui, st),
                        Section::Audio => audio(ui, st),
                        Section::System => system(ui, st),
                    });
                    ui.separator();
                    if let Some(e) = &st.error {
                        ui.colored_label(Color32::from_rgb(255, 110, 110), e);
                    } else if let Some(s) = &st.status {
                        ui.label(RichText::new(s).small());
                    }
                    ui.label(
                        RichText::new("Controller: D-pad / left stick move, Cross select, Circle back, L1 / R1 sections, Options or PS closes. Keyboard: arrows, Enter, Esc.")
                            .small()
                            .weak(),
                    );
                });
            });
        });
}

/// The binding set the hint shows for a device mode.
pub fn hint_set(mode: DeviceMode) -> BindingSet {
    crate::controls_overlay::set_for(mode)
}

/// The read-only controls hint over the game.
fn draw_hint(ctx: &egui::Context, st: &mut MenuState) {
    let feed = ps5input::menu::controllers();
    if st.input.is_none() {
        st.hint = false;
        return;
    }
    let mut cfg = st.effective_input();
    // Tracks the held keys and controller inputs (no capture runs here).
    st.bindings.capture = None;
    st.bindings.feed(ctx, feed.pressed, &mut cfg);
    let screen = ctx.screen_rect();
    egui::Area::new(egui::Id::new("hint-backdrop")).order(egui::Order::Background).fixed_pos(screen.min).show(ctx, |ui| {
        ui.painter().rect_filled(screen, 0.0, Color32::from_black_alpha(110));
        ui.allocate_rect(screen, egui::Sense::click());
    });
    let width = (screen.width() - 32.0).clamp(420.0, 1100.0);
    egui::Window::new(RichText::new("Controls").strong())
        .id(egui::Id::new("controls-hint"))
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .fixed_size([width, 0.0])
        .frame(egui::Frame::window(&ctx.style()).fill(Color32::from_rgba_unmultiplied(24, 26, 30, 235)))
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                st.bindings.set_tabs(ui);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Close").clicked() {
                        st.close_request = true;
                    }
                    if ui.button("Edit").on_hover_text("Rebind in the menu's Controls page").clicked() {
                        st.hint = false;
                        st.section = Section::Controls;
                    }
                });
            });
            ui.label(RichText::new(crate::controls_overlay::source_line(&cfg, &st.bindings_source())).small().weak());
            let set = st.bindings.set;
            let (held, pad) = st.bindings.held(true);
            let pressed = dualsense::pressed_controls(&cfg, set, &held, &pad);
            let params = dualsense::PictureParams {
                cfg: &cfg,
                set,
                selected: None,
                capturing: false,
                pressed: &pressed,
                interactive: false,
                max_width: width,
            };
            dualsense::picture(ui, &params);
            let mut note = "Held keys and buttons light up. Esc, the menu key, Options or PS close.".to_string();
            if let Some(k) = crate::controls_overlay::emulator_key(&cfg, Control::ShowControls) {
                note.push_str(&format!(" In game, hold {k} for every binding over the running game."));
            }
            ui.label(RichText::new(note).small().weak());
        });
    ctx.request_repaint_after(std::time::Duration::from_millis(16));
}

fn confirm_exit(ui: &mut egui::Ui, st: &mut MenuState) {
    egui::Frame::group(ui.style()).fill(Color32::from_rgb(60, 30, 30)).show(ui, |ui| {
        ui.label(RichText::new("Exit the game? Unsaved game progress is lost.").strong());
        ui.horizontal(|ui| {
            let yes = ui.button(if st.from_launcher { "Exit to the launcher" } else { "Exit and open the launcher" });
            if std::mem::take(&mut st.focus_confirm) {
                yes.request_focus();
            }
            if yes.clicked() {
                st.flush();
                exit_to_launcher(st.from_launcher);
            }
            if ui.button("Cancel").clicked() {
                st.confirm_exit = false;
            }
        });
    });
}

fn after_restart(ui: &mut egui::Ui) {
    ui.label(RichText::new("after restart").small().color(Color32::from_rgb(220, 180, 90)));
}

fn controls(ui: &mut egui::Ui, st: &mut MenuState) {
    let feed = ps5input::menu::controllers();
    let Some(store) = st.input.as_ref() else {
        ui.label("Input settings are unavailable (no configuration folder).");
        return;
    };
    if ui.button("Show the controls hint").on_hover_text("The controller picture over the game, read-only").clicked() {
        st.open_hint();
        return;
    }
    let mut profile = store.profile_enabled;
    let game = matches!(store.target, Target::Game { .. });
    let editable = store.editable();
    let mut cfg = store.config.clone();
    let path = store.path();
    if game {
        if ui.checkbox(&mut profile, "Use a per-game input profile for this game").changed() {
            st.set_profile_enabled(profile);
        }
        if !editable {
            ui.label(RichText::new("This game uses the global input settings; turn its profile on to change them here.").italics());
        }
    } else {
        ui.label(RichText::new("No title id: the global input settings are edited.").italics());
    }
    if st.bindings.feed(ui.ctx(), feed.pressed.clone(), &mut cfg) {
        st.status = None;
    }
    ui.add_enabled_ui(editable, |ui| {
        input_ui::device_settings(ui, &mut cfg, &feed.devices, feed.error.as_deref(), false);
        egui::CollapsingHeader::new("Controller: light bar, speaker, vibration").show(ui, |ui| input_ui::controller_settings(ui, &mut cfg));
        egui::CollapsingHeader::new("Analog and mouse").show(ui, |ui| input_ui::analog_settings(ui, &mut cfg));
        egui::CollapsingHeader::new("Key and button bindings").default_open(true).show(ui, |ui| st.bindings.show(ui, &mut cfg));
        ui.horizontal(|ui| {
            if ui.button("Undo changes").on_hover_text("Back to the settings when the menu opened").clicked() {
                if let Some(c) = st.input_at_open.clone() {
                    cfg = c;
                }
            }
            if ui.button("Reset to defaults").clicked() {
                cfg = Config::default();
            }
        });
    });
    ui.label(RichText::new(format!("Saved to {}", path.display())).small().weak());
    if editable {
        st.set_input(cfg);
    }
}

fn combo(ui: &mut egui::Ui, id: &str, value: &mut String, options: &[(&str, &str)]) {
    let current = options.iter().find(|o| o.0 == value.as_str()).map(|o| o.1).unwrap_or("?");
    egui::ComboBox::from_id_salt(id).selected_text(current).width(260.0).show_ui(ui, |ui| {
        for (k, label) in options {
            ui.selectable_value(value, k.to_string(), *label);
        }
    });
}

fn graphics(ui: &mut egui::Ui, st: &mut MenuState) {
    let mut s = st.settings.clone();
    egui::Grid::new("menu-gfx").num_columns(3).spacing([16.0, 10.0]).show(ui, |ui| {
        ui.label("Present mode");
        combo(ui, "menu-present", &mut s.present_mode, &config::PRESENT_MODES);
        ui.end_row();
        ui.label("Fullscreen");
        ui.checkbox(&mut s.fullscreen, "Fullscreen window (F11)");
        ui.end_row();
        ui.label("HDR");
        ui.horizontal(|ui| {
            ui.checkbox(&mut s.hdr, "HDR10 output (needs an HDR display)");
            if s.hdr != st.started.hdr {
                after_restart(ui);
            }
        });
        ui.end_row();
        ui.label("Display resolution");
        ui.horizontal(|ui| {
            combo(ui, "menu-display-res", &mut s.display_resolution, &config::DISPLAY_RESOLUTIONS);
            if s.display_resolution != st.started.display_resolution {
                after_restart(ui);
            }
        });
        ui.end_row();
        ui.label("Scaling");
        combo(ui, "menu-scale", &mut s.scale_mode, &config::SCALE_MODES);
        ui.end_row();
        ui.label("Filter");
        combo(ui, "menu-filter", &mut s.scale_filter, &config::SCALE_FILTERS);
        ui.end_row();
        ui.label("Performance overlay");
        ui.checkbox(&mut s.perf_overlay, "FPS, frame times, CPU / GPU load (F10)");
        ui.end_row();
        ui.label("FPS counter");
        ui.checkbox(&mut s.fps_counter, "Frames per second in the corner");
        ui.end_row();
        ui.label("GPU fast paths");
        ui.checkbox(&mut s.fast_paths, "Per-draw fast paths (memoised bindings, draw recipes)");
        ui.end_row();
        ui.label("Guest refresh rate");
        ui.horizontal(|ui| {
            let mut custom = s.refresh_rate != 0.0;
            if ui.checkbox(&mut custom, "custom").changed() {
                s.refresh_rate = if custom { 60.0 } else { 0.0 };
            }
            if custom {
                ui.add(egui::DragValue::new(&mut s.refresh_rate).range(30.0..=360.0).speed(0.5).suffix(" Hz"));
            } else {
                ui.label("59.94 Hz (frame rate cap)");
            }
            after_restart(ui);
        });
        ui.end_row();
    });
    egui::CollapsingHeader::new("GPU switches (after restart)").show(ui, |ui| {
        for (i, sw) in GPU_SWITCHES.iter().enumerate() {
            let mut on = sw.on_in(&s.extra_env);
            ui.horizontal(|ui| {
                if ui.checkbox(&mut on, sw.label).on_hover_text(sw.env).changed() {
                    sw.set_in(&mut s.extra_env, on);
                }
                if st.started.switches.get(i).is_some_and(|&was| was != on) {
                    after_restart(ui);
                }
            });
        }
    });
    st.set_settings(s);
}

fn audio(ui: &mut egui::Ui, st: &mut MenuState) {
    let mut s = st.settings.clone();
    egui::Grid::new("menu-audio").num_columns(2).spacing([16.0, 10.0]).show(ui, |ui| {
        ui.label("Volume");
        ui.add(egui::Slider::new(&mut s.volume, 0..=100).suffix(" %"));
        ui.end_row();
        ui.label("");
        ui.label(RichText::new("The controller speaker volume is under Controls.").small().weak());
        ui.end_row();
    });
    st.set_settings(s);
}

fn system(ui: &mut egui::Ui, st: &mut MenuState) {
    let mut s = st.settings.clone();
    egui::Grid::new("menu-sys").num_columns(2).spacing([16.0, 10.0]).show(ui, |ui| {
        ui.label("Console language");
        ui.horizontal(|ui| {
            let current = config::LANGUAGES.iter().find(|l| l.0 == s.console_language).map(|l| l.1).unwrap_or("?");
            egui::ComboBox::from_id_salt("menu-lang").selected_text(current).width(220.0).show_ui(ui, |ui| {
                for (id, name) in config::LANGUAGES {
                    ui.selectable_value(&mut s.console_language, *id, *name);
                }
            });
            if s.console_language != st.started.console_language {
                after_restart(ui);
            }
        });
        ui.end_row();
        ui.label("Pause");
        ui.checkbox(&mut s.pause_in_menu, "Pause the game while this menu is open");
        ui.end_row();
        ui.label("Shaders");
        ui.horizontal(|ui| {
            let busy = ps5gpu::pipeline::precompile::recompiling();
            let b = ui
                .add_enabled(!busy, egui::Button::new("Recompile shaders"))
                .on_hover_text("Clear this game's shader and pipeline caches and build them again on every core (the game is paused meanwhile)");
            if b.clicked() {
                st.recompile_request = true;
                st.close_request = true;
            }
            if busy {
                ui.label(RichText::new("running").small());
            }
        });
        ui.end_row();
        ui.label("Game");
        ui.label(format!("{} ({})", st.game.title, st.game.title_id.as_deref().unwrap_or("no title id")));
        ui.end_row();
        ui.label("Configuration");
        let where_ = if st.own_config { "the game's own configuration" } else { "global settings (saving creates the game's configuration)" };
        ui.label(RichText::new(where_).small());
        ui.end_row();
    });
    st.set_settings(s);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ps5gpu::menu::MenuApp;
    use ps5input::button;

    fn dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("rps5e-gui-ingame-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn game() -> GameInfo {
        GameInfo { key: "/games/SH".into(), title: "SILENT HILL".into(), title_id: Some("PPSA10112".into()) }
    }

    fn started() -> Started {
        Started::from_vars(|_| None)
    }

    #[test]
    fn sections_cycle() {
        assert_eq!(Section::Controls.next(), Section::Graphics);
        assert_eq!(Section::Controls.prev(), Section::System);
        assert_eq!(Section::System.next(), Section::Controls);
    }

    #[test]
    fn gpu_switches_in_the_environment_list() {
        let safe = GPU_SWITCHES.iter().find(|s| s.env == "RPS5E_SAFE").unwrap();
        let memo = GPU_SWITCHES.iter().find(|s| s.env == "RPS5E_VIEW_MEMO").unwrap();
        let mut env = vec!["RPS5E_GPU_TRACE=1".to_string()];
        assert!(!safe.on_in(&env) && memo.on_in(&env));
        safe.set_in(&mut env, true);
        memo.set_in(&mut env, false);
        assert_eq!(env, ["RPS5E_GPU_TRACE=1", "RPS5E_SAFE=1", "RPS5E_VIEW_MEMO=0"]);
        assert!(safe.on_in(&env) && !memo.on_in(&env));
        // Back to the defaults: nothing is left in the list.
        safe.set_in(&mut env, false);
        memo.set_in(&mut env, true);
        assert_eq!(env, ["RPS5E_GPU_TRACE=1"]);
        assert!(!memo.is_on(Some("off")) && memo.is_on(Some("1")) && !safe.is_on(Some("yes")));
    }

    #[test]
    fn live_and_restart_settings() {
        let a = EmulatorSettings::default();
        let mut b = a.clone();
        b.present_mode = "fifo".into();
        b.volume = 30;
        b.scale_mode = "integer".into();
        b.fps_counter = true;
        b.console_language = 8;
        assert_eq!(
            live_changes(&a, &b),
            [
                LiveChange::PresentMode(PresentMode::Fifo),
                LiveChange::FpsCounter(true),
                LiveChange::ScaleMode(ScaleMode::Integer),
                LiveChange::Volume(30)
            ]
        );
        assert!(live_changes(&a, &a).is_empty());
        let st = started();
        assert_eq!(restart_pending(&b, &st), ["console language"]);
        GPU_SWITCHES[0].set_in(&mut b.extra_env, true);
        b.refresh_rate = 120.0;
        assert_eq!(restart_pending(&b, &st), ["console language", "refresh rate", GPU_SWITCHES[0].env]);
        // HDR is read at start too; it is not a live change.
        b.hdr = true;
        assert_eq!(live_changes(&a, &b).len(), 4);
        assert_eq!(restart_pending(&b, &st), ["console language", "refresh rate", "HDR", GPU_SWITCHES[0].env]);
        // So is the display resolution.
        b.display_resolution = "1080p".into();
        assert_eq!(live_changes(&a, &b).len(), 4);
        assert_eq!(
            restart_pending(&b, &st),
            ["console language", "refresh rate", "HDR", "display resolution", GPU_SWITCHES[0].env]
        );
        // What the game started with counts.
        let st = Started::from_vars(|k| match k {
            "RPS5E_SYSTEM_LANGUAGE" => Some("8".into()),
            "RPS5E_GUEST_REFRESH_RATE" => Some("120".into()),
            "RPS5E_HDR" => Some("1".into()),
            "RPS5E_DISPLAY_RESOLUTION" => Some("1080P".into()),
            "RPS5E_SAFE" => Some("1".into()),
            _ => None,
        });
        assert!(st.hdr);
        assert!(restart_pending(&b, &st).is_empty());
        assert!(!started().hdr, "HDR is off without RPS5E_HDR");
        assert_eq!(started().display_resolution, "2160p", "2160p without RPS5E_DISPLAY_RESOLUTION");
    }

    #[test]
    fn settings_are_saved_to_the_game_configuration() {
        let d = dir("save");
        // No game configuration: the global settings are the start.
        let mut global = GuiSettings::default();
        global.emulator.log_level = "debug".into();
        global.save_to(&d.join(crate::settings::FILE_NAME)).unwrap();
        let mut st = MenuState::new(game(), Some(d.clone()), Some(d.clone()), started());
        assert!(!st.own_config);
        assert_eq!(st.settings.log_level, "debug");
        let mut next = st.settings.clone();
        next.console_language = 8;
        GPU_SWITCHES[1].set_in(&mut next.extra_env, false);
        st.set_settings(next);
        st.flush();
        assert!(st.own_config && st.error.is_none(), "{:?}", st.error);
        // The launcher's file has the game's configuration now.
        let configs = GameConfigs::load(&d).unwrap();
        let g = &configs.games["/games/SH"];
        assert_eq!((g.title_id.as_str(), g.name.as_str()), ("PPSA10112", "SILENT HILL"));
        assert_eq!(g.settings.console_language, 8);
        assert_eq!(g.settings.log_level, "debug");
        assert!(!GPU_SWITCHES[1].on_in(&g.settings.extra_env));
        // ... and the next start's environment carries it.
        let env = g.settings.env(None, None, None);
        assert!(env.contains(&("RPS5E_SYSTEM_LANGUAGE".into(), "8".into())));
        assert!(env.contains(&(GPU_SWITCHES[1].env.into(), "0".into())));
        // Reopened: the game's own configuration is loaded.
        let st = MenuState::new(game(), Some(d.clone()), Some(d.clone()), started());
        assert!(st.own_config);
        assert_eq!(st.settings.console_language, 8);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn input_changes_go_to_the_game_profile() {
        let d = dir("input");
        let mut st = MenuState::new(game(), Some(d.clone()), Some(d.clone()), started());
        assert!(!st.input.as_ref().unwrap().editable(), "no profile yet: the global settings apply");
        st.set_profile_enabled(true);
        let mut cfg = st.input.as_ref().unwrap().config.clone();
        cfg.mode = ps5input::mapping::DeviceMode::Keyboard;
        cfg.set_sources(Control::Button(button::CROSS), &[Source::Key(ps5gpu::input::KeyCode::KeyZ)], true).unwrap();
        st.set_input(cfg);
        st.flush();
        assert!(st.error.is_none(), "{:?}", st.error);
        // What the emulator reads at the next start.
        let profiles = d.join(ps5input::profile::PROFILES_DIR);
        let r = ps5input::profile::resolve_with(None, Some(&profiles), Some("PPSA10112")).unwrap();
        assert_eq!(r.config.mode, ps5input::mapping::DeviceMode::Keyboard);
        assert_eq!(r.config.sources_of(Control::Button(button::CROSS), true), [Source::Key(ps5gpu::input::KeyCode::KeyZ)]);
        // The menu also closes with its own key (F1 by default).
        assert!(st.menu_keys().contains(&ps5gpu::input::KeyCode::F1));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn controller_navigation_drives_the_menu() {
        let d = dir("nav");
        let mut app = InGameMenu::new(MenuState::new(game(), Some(d.clone()), Some(d.clone()), started()));
        let ctx = egui::Context::default();
        let frame = |app: &mut InGameMenu, steps: &[Nav]| {
            ps5input::menu::publish(Vec::new(), Vec::new(), steps);
            let mut raw = egui::RawInput::default();
            app.raw_input(&mut raw);
            ctx.run(raw, |ctx| app.ui(ctx))
        };
        JUST_OPENED.store(true, Ordering::Release);
        frame(&mut app, &[]);
        let out = frame(&mut app, &[]);
        assert!(!out.shapes.is_empty(), "the menu draws");
        assert_eq!(app.state.section, Section::Controls);
        frame(&mut app, &[Nav::NextTab]);
        assert_eq!(app.state.section, Section::Graphics);
        frame(&mut app, &[Nav::PrevTab, Nav::PrevTab]);
        assert_eq!(app.state.section, Section::System);
        // Directions and Cross become keys (egui moves the focus with them).
        let mut raw = egui::RawInput::default();
        ps5input::menu::publish(Vec::new(), Vec::new(), &[Nav::Down, Nav::Accept]);
        app.raw_input(&mut raw);
        let keys: Vec<_> = raw
            .events
            .iter()
            .filter_map(|e| match e {
                egui::Event::Key { key, pressed: true, .. } => Some(*key),
                _ => None,
            })
            .collect();
        assert_eq!(keys, [egui::Key::ArrowDown, egui::Key::Enter]);
        // While a binding is captured the controller is no navigation.
        app.state.bindings.start(ps5input::capture::CaptureTarget::Gamepad(button::CROSS), false, std::time::Instant::now());
        let mut raw = egui::RawInput::default();
        ps5input::menu::publish(Vec::new(), Vec::new(), &[Nav::Down, Nav::Close]);
        app.raw_input(&mut raw);
        assert!(raw.events.is_empty() && app.actions == NavActions::default());
        app.state.bindings.capture = None;
        // Options closes the menu.
        frame(&mut app, &[Nav::Close]);
        assert!(!ps5gpu::menu::is_open());
        let _ = std::fs::remove_dir_all(&d);
    }

    fn key_event(k: egui::Key, pressed: bool) -> egui::Event {
        egui::Event::Key { key: k, physical_key: Some(k), pressed, repeat: false, modifiers: egui::Modifiers::NONE }
    }

    #[test]
    fn controls_hint_shows_and_closes() {
        assert_eq!(hint_set(DeviceMode::Controller), BindingSet::Controller);
        assert_eq!(hint_set(DeviceMode::Keyboard), BindingSet::Keyboard);
        assert_eq!(hint_set(DeviceMode::Auto), BindingSet::KeyboardMouse);
        let d = dir("hint");
        let mut app = InGameMenu::new(MenuState::new(game(), Some(d.clone()), Some(d.clone()), started()));
        app.state.bindings.set = BindingSet::Controller;
        app.state.open_hint();
        assert!(app.state.hint);
        assert_eq!(app.state.bindings.set, BindingSet::KeyboardMouse, "the set of the device mode (auto)");
        let ctx = egui::Context::default();
        let out = ctx.run(egui::RawInput::default(), |c| app.ui(c));
        assert!(!out.shapes.is_empty(), "the hint draws");
        // A held key is tracked (its control lights up), the hint stays.
        let raw = egui::RawInput { events: vec![key_event(egui::Key::Space, true)], ..Default::default() };
        let _ = ctx.run(raw, |c| app.ui(c));
        assert!(app.state.hint);
        assert_eq!(app.state.bindings.held(true).0, [Source::Key(ps5gpu::input::KeyCode::Space)]);
        // Escape closes the hint and the menu with it.
        let raw = egui::RawInput { events: vec![key_event(egui::Key::Escape, true)], ..Default::default() };
        let _ = ctx.run(raw, |c| app.ui(c));
        assert!(!app.state.hint);
        let _ = std::fs::remove_dir_all(&d);
    }

    /// `RPS5E_GUI_RENDER_DIR=<dir> cargo test -p rps5e-gui render_ingame -- --ignored`
    /// writes the in-game Controls page, the controls hint and the Graphics
    /// page as PNG files.
    #[test]
    #[ignore]
    fn render_ingame() {
        let Some(out) = std::env::var_os("RPS5E_GUI_RENDER_DIR") else { return };
        let out = PathBuf::from(out);
        std::fs::create_dir_all(&out).unwrap();
        let d = dir("render");
        let mut app = InGameMenu::new(MenuState::new(game(), Some(d.clone()), Some(d.clone()), started()));
        let mut r = crate::softrender::SoftRenderer::new([1280.0, 720.0], 1.0);
        let mut shot = |app: &mut InGameMenu, events: Vec<egui::Event>, name: &str| {
            let mut img = (0, 0, Vec::new());
            for ev in std::iter::once(events).chain((0..30).map(|_| Vec::new())) {
                img = r.frame(ev, |ctx| app.ui(ctx));
            }
            crate::softrender::write_png(&out.join(name), img.0, img.1, &img.2);
        };
        shot(&mut app, Vec::new(), "ingame-controls-global.png");
        // With the game's own profile the page is editable.
        app.state.set_profile_enabled(true);
        shot(&mut app, Vec::new(), "ingame-controls.png");
        app.state.open_hint();
        shot(&mut app, vec![key_event(egui::Key::Space, true), key_event(egui::Key::W, true)], "ingame-hint.png");
        // Graphics with a display resolution changed after the start.
        app.state.hint = false;
        app.state.section = Section::Graphics;
        let mut next = app.state.settings.clone();
        next.display_resolution = "1080p".into();
        app.state.set_settings(next);
        shot(&mut app, Vec::new(), "ingame-graphics.png");
        let _ = std::fs::remove_dir_all(&d);
    }

    /// Keys as egui-winit sends them: a key event with its physical key,
    /// then the typed text.
    fn typed(k: egui::Key, text: &str) -> Vec<egui::Event> {
        vec![key_event(k, true), egui::Event::Text(text.into())]
    }

    #[test]
    fn letter_key_rebinds_after_the_menu_was_closed_with_a_key() {
        let d = dir("rebind");
        let mut app = InGameMenu::new(MenuState::new(game(), Some(d.clone()), Some(d.clone()), started()));
        let ctx = egui::Context::default();
        // One frame; `opened`: the first after the menu opened.
        let frame = |app: &mut InGameMenu, events: Vec<egui::Event>, opened: bool| {
            let mut raw = egui::RawInput { events, ..Default::default() };
            if opened {
                app.opened(&mut raw);
            } else {
                app.raw_input(&mut raw);
            }
            let _ = ctx.run(raw, |c| app.ui(c));
        };
        let run = |app: &mut InGameMenu, events: Vec<egui::Event>| frame(app, events, false);
        app.state.set_profile_enabled(true);
        frame(&mut app, Vec::new(), true);
        // Escape closes the menu; its release goes to the game, not to egui.
        run(&mut app, vec![key_event(egui::Key::Escape, true)]);
        assert!(ctx.input(|i| i.keys_down.contains(&egui::Key::Escape)), "egui still holds the key");
        frame(&mut app, Vec::new(), true);
        assert!(ctx.input(|i| i.keys_down.is_empty()), "released when the menu opens");
        let cross = Control::Button(button::CROSS);
        for (k, text, code) in [
            (egui::Key::X, "x", ps5gpu::input::KeyCode::KeyX),
            (egui::Key::C, "c", ps5gpu::input::KeyCode::KeyC),
            (egui::Key::Z, "\u{44f}", ps5gpu::input::KeyCode::KeyZ),
            (egui::Key::Num4, "4", ps5gpu::input::KeyCode::Digit4),
            (egui::Key::Period, ".", ps5gpu::input::KeyCode::Period),
        ] {
            app.state.bindings.start(ps5input::capture::CaptureTarget::KeyboardMouse(cross), false, std::time::Instant::now());
            run(&mut app, Vec::new());
            run(&mut app, typed(k, text));
            assert!(!app.state.bindings.capturing(), "{k:?} ends the capture");
            assert_eq!(app.state.input.as_ref().unwrap().config.sources_of(cross, false), [Source::Key(code)]);
            run(&mut app, vec![key_event(k, false)]);
        }
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn nav_keys() {
        assert_eq!(nav_key(Nav::Left), Some(egui::Key::ArrowLeft));
        assert_eq!(nav_key(Nav::Accept), Some(egui::Key::Enter));
        assert_eq!(nav_key(Nav::Back), None);
    }
}
