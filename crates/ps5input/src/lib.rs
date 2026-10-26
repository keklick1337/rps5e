// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Host input: game controllers (SDL3), keyboard and mouse of the
//! presentation window, mapped onto the emulated pads (libScePad, up to
//! four local players) and, when enabled, the guest mouse (libSceMouse).
//!
//! [`start`] resolves the settings (`rps5e-input.toml` plus the title's
//! per-game profile, see [`mapping`] and [`profile`]), signs in one user
//! per configured player, hooks the window's input sink, installs the
//! rumble/light bar sink and runs an input thread that polls controllers
//! every 4 ms and (re)assigns them to players on hot-plug. Player 1's
//! pad state follows the device mode: controller, keyboard and/or mouse
//! bindings overlaid on the controller snapshot. `RPS5E_NO_HOST_INPUT=1`
//! disables all of it.
//!
//! `RPS5E_PAD_SCRIPT="12:cross,20:down,21:cross:400"` (unattended debug runs)
//! taps the named buttons at the given seconds after start, held for the
//! optional third field in ms (150 ms by default); `lup`/`ldown`/`lleft`/`lright`
//! (and `r...`) push a stick fully, `a+b` presses buttons together;
//! it works with host input disabled.
//!
//! While the in-game menu (`ps5gpu::menu`) is open the guest sees neutral
//! pads and player 1's controller navigates the menu ([`menu`]);
//! [`reconfigure`] applies settings edited there to the running game.

pub mod assign;
pub mod capture;
pub mod haptics;
pub mod mapping;
pub mod motion;
pub mod menu;
pub mod output;
pub mod profile;
pub mod sdl;
pub mod shaping;
pub mod state;
pub mod touch;

pub use ps5hle::pad::{button, HostBattery, HostPadState};

use mapping::{Config, Control, DeviceMode, MAX_PLAYERS};
use ps5gpu::input::{request_window, show_hint, InputEvent, MouseButton, WindowRequest};
use ps5hle::pad::{self, mouse_button};
use state::{Effects, KbmState};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

const POLL_INTERVAL: Duration = Duration::from_millis(4);
const HINT_TIME: Duration = Duration::from_secs(5);

struct Hub {
    kbm: KbmState,
    mode: DeviceMode,
    players: usize,
    /// Latest controller snapshot per player (`None`: no device).
    pads: [Option<HostPadState>; MAX_PLAYERS],
    guest_mouse: bool,
    mouse_buttons: u32,
    /// Mouse-look and touch pad mode were on when the menu opened
    /// (restored when it closes).
    menu_look: bool,
    menu_touch: bool,
}

static HUB: OnceLock<Mutex<Hub>> = OnceLock::new();
/// Settings from [`reconfigure`] for the input thread.
static PENDING: Mutex<Option<Config>> = Mutex::new(None);
/// The settings in effect (resolved at start, replaced by [`reconfigure`]).
static ACTIVE: Mutex<Option<Config>> = Mutex::new(None);

/// The input settings the running game uses: the per-game profile when it
/// is on, else the global file, with the command-line overrides. `None`
/// before host input started or with it disabled.
pub fn active_config() -> Option<Config> {
    ACTIVE.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

fn set_active(cfg: &Config) {
    *ACTIVE.lock().unwrap_or_else(|e| e.into_inner()) = Some(cfg.clone());
}

fn hub() -> Option<MutexGuard<'static, Hub>> {
    HUB.get().map(|m| m.lock().unwrap_or_else(|e| e.into_inner()))
}

impl Hub {
    fn publish(&self) {
        if ps5gpu::menu::is_open() {
            // The menu has the controls: the game sees released pads.
            pad::set_host_pad_state(HostPadState::NEUTRAL);
            for p in 1..self.players {
                let connected = self.pads[p].is_some();
                pad::set_host_pad_state_for(p, HostPadState { connected, ..HostPadState::NEUTRAL });
            }
            return;
        }
        // Player 1: the keyboard/mouse bindings of its mode overlaid on its
        // controller (neutral without one).
        let p1 = self.pads[0].unwrap_or(HostPadState::NEUTRAL);
        pad::set_host_pad_state(self.kbm.compose(&p1));
        for p in 1..self.players {
            let state = self.pads[p].map(|s| HostPadState { connected: true, ..s });
            pad::set_host_pad_state_for(p, state.unwrap_or(HostPadState { connected: false, ..HostPadState::NEUTRAL }));
        }
    }
}

fn guest_mouse_bit(b: MouseButton) -> u32 {
    match b {
        MouseButton::Left => mouse_button::PRIMARY,
        MouseButton::Right => mouse_button::SECONDARY,
        MouseButton::Middle => mouse_button::OPTIONAL,
        MouseButton::X1 => mouse_button::OPTIONAL2,
        MouseButton::X2 => mouse_button::OPTIONAL3,
        MouseButton::Other(_) => 0,
    }
}

fn sign(v: f32) -> i32 {
    if v > 0.0 {
        1
    } else if v < 0.0 {
        -1
    } else {
        0
    }
}

fn key_label(k: &KbmState, control: Control) -> Option<String> {
    k.key_for(control).and_then(mapping::key_name)
}

/// On-screen hint after a mouse capture change.
fn capture_hint(k: &KbmState, on: bool) -> String {
    if on {
        let mut keys: Vec<String> = [Control::ToggleMouse, Control::ReleaseMouse].iter().filter_map(|&c| key_label(k, c)).collect();
        keys.dedup();
        if keys.is_empty() {
            "Mouse captured (mouse-look)".to_string()
        } else {
            format!("Mouse captured (mouse-look) - {} to release", keys.join(" / "))
        }
    } else {
        match key_label(k, Control::ToggleMouse) {
            Some(key) => format!("Mouse released - {key} to capture"),
            None => "Mouse released".to_string(),
        }
    }
}

/// On-screen hint after touch pad mode changed.
fn touch_mode_hint(k: &KbmState, on: bool) -> String {
    let key = key_label(k, Control::TouchpadMode);
    if on {
        let back = key.map(|k| format!(" - {k} to leave")).unwrap_or_default();
        format!("Touch pad mode: numpad 1-9 touch, left mouse button touches, right one clicks{back}")
    } else {
        "Touch pad mode off: the numpad tilts the controller".to_string()
    }
}

/// Where Create saves a screenshot: `screenshots/<title>-<ms>.png` in the
/// user data folder.
fn screenshot_path() -> std::path::PathBuf {
    let title = ps5hle::app_info::title_id().unwrap_or_else(|| "game".into());
    let safe: String = title.chars().filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_').collect();
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0);
    ps5hle::filesystem::user_data_dir().join("screenshots").join(format!("{safe}-{now:015}.png"))
}

/// Hint shown when the game window opens.
fn startup_hint(k: &KbmState, mode: DeviceMode) -> Option<String> {
    let mut parts = Vec::new();
    if mode.uses_mouse() {
        if let Some(key) = key_label(k, Control::ToggleMouse) {
            parts.push(format!("{key}: capture mouse"));
        }
    }
    if let Some(key) = key_label(k, Control::ToggleFullscreen) {
        parts.push(format!("{key}: fullscreen"));
    }
    if let Some(key) = key_label(k, Control::TogglePerfOverlay) {
        parts.push(format!("{key}: performance"));
    }
    if ps5gpu::menu::installed() {
        if let Some(key) = key_label(k, Control::ToggleMenu) {
            parts.push(format!("{key}: menu"));
        }
        if let Some(key) = key_label(k, Control::ShowControls) {
            parts.push(format!("{key} (hold): controls"));
        }
    }
    (!parts.is_empty()).then(|| parts.join("   "))
}

/// Window input sink (render thread).
fn on_event(ev: InputEvent) {
    let Some(mut h) = hub() else { return };
    let mouse = h.mode.uses_mouse();
    let fx = match ev {
        InputEvent::Key { code, pressed, repeat } => h.kbm.key(code, pressed, repeat),
        // Keyboard-only and controller modes ignore the mouse entirely.
        InputEvent::MouseButton { .. } | InputEvent::MouseMotion { .. } | InputEvent::Wheel { .. } if !mouse => {
            return;
        }
        InputEvent::MouseButton { button, pressed } => {
            if h.guest_mouse {
                let bit = guest_mouse_bit(button);
                h.mouse_buttons = if pressed { h.mouse_buttons | bit } else { h.mouse_buttons & !bit };
                pad::push_host_mouse_event(0, 0, 0, 0, h.mouse_buttons);
            }
            h.kbm.mouse_button(button, pressed)
        }
        // While the title reads the mouse itself, motion and the wheel do
        // not also drive the pad bindings (mouse-look, wheel binds).
        InputEvent::MouseMotion { dx, dy } => {
            if !(h.guest_mouse && pad::mouse_is_open()) {
                h.kbm.mouse_motion(dx, dy);
            }
            if h.guest_mouse {
                pad::push_host_mouse_event(dx.round() as i32, dy.round() as i32, 0, 0, h.mouse_buttons);
            }
            Effects::default()
        }
        InputEvent::Wheel { dx, dy } => {
            if !(h.guest_mouse && pad::mouse_is_open()) {
                h.kbm.wheel(dy, Instant::now());
            }
            if h.guest_mouse {
                pad::push_host_mouse_event(0, 0, sign(dy), sign(dx), h.mouse_buttons);
            }
            Effects::default()
        }
        InputEvent::Focus(true) => Effects::default(),
        InputEvent::Focus(false) => {
            if h.guest_mouse && h.mouse_buttons != 0 {
                h.mouse_buttons = 0;
                pad::push_host_mouse_event(0, 0, 0, 0, 0);
            }
            h.kbm.release_all()
        }
    };
    h.publish();
    let mut hint = fx.capture_mouse.map(|on| capture_hint(&h.kbm, on));
    // Touch pad mode and held mouse motion capture the mouse while they
    // last (mouse modes only).
    let mut capture = fx.capture_mouse;
    if let Some(on) = fx.touch_mode {
        hint = Some(touch_mode_hint(&h.kbm, on));
        if mouse && fx.capture_mouse.is_none() {
            capture = Some(on || h.kbm.mouse_look);
        }
    }
    if let (Some(on), true) = (fx.hold_capture, mouse) {
        capture = Some(on);
    }
    drop(h);
    if fx.screenshot {
        let path = screenshot_path();
        hint = Some(format!("Screenshot: {}", path.display()));
        ps5gpu::request_screenshot(path);
    }
    if fx.toggle_mute {
        let off = ps5hle::audio::host::toggle_user_muted();
        hint = Some(if off { "Sound muted".to_string() } else { "Sound on".to_string() });
    }
    if fx.toggle_fullscreen {
        request_window(WindowRequest::ToggleFullscreen);
    }
    if fx.toggle_perf_overlay {
        ps5gpu::perf::toggle_overlay();
    }
    if fx.toggle_menu {
        ps5gpu::menu::toggle();
    }
    if let Some(on) = fx.show_controls {
        ps5gpu::menu::set_overlay(on);
    }
    if let Some(setting) = fx.cycle {
        pad::cycle_setting(setting);
    }
    if let Some(on) = capture {
        request_window(WindowRequest::CaptureMouse(on));
    }
    if let Some(text) = hint {
        show_hint(text, HINT_TIME);
    }
}

/// Resolve the settings for the running title (global file + profile).
/// TOML applied on top of the resolved settings (command-line overrides,
/// see `rps5e-cli run --bind/--input-mode`).
pub const OVERRIDE_ENV: &str = "RPS5E_INPUT_OVERRIDE";

fn resolve_settings() -> Config {
    with_overrides(resolve_file_settings(), true)
}

/// `cfg` with the command-line overrides on top.
fn with_overrides(cfg: Config, log_it: bool) -> Config {
    match std::env::var(OVERRIDE_ENV) {
        Ok(text) if !text.trim().is_empty() => match mapping::parse_config_onto(cfg.clone(), &text) {
            Ok(c) => {
                if log_it {
                    log::info!("input: command-line overrides applied");
                }
                c
            }
            Err(e) => {
                log::error!("input: bad command-line input override: {e}");
                cfg
            }
        },
        _ => cfg,
    }
}

/// Apply edited settings to the running game (the in-game menu): bindings,
/// device mode, controller choice and layout, sticks, mouse and controller
/// features. The command-line overrides stay on top. The number of local
/// players is fixed at start.
pub fn reconfigure(cfg: &Config) {
    let mut cfg = with_overrides(cfg.clone(), false);
    let mut capture = None;
    if let Some(mut h) = hub() {
        cfg.players = h.players;
        let (look, touch) = (h.kbm.mouse_look, h.kbm.touch_mode);
        h.kbm = KbmState::from_config(&cfg);
        h.mode = cfg.mode;
        if touch {
            h.kbm.set_touch_mode(true);
        }
        if !cfg.mode.uses_mouse() {
            h.menu_look = false;
            if look || touch {
                capture = Some(false);
            }
        } else if look {
            h.kbm.set_mouse_look(true);
        }
        h.guest_mouse = cfg.guest_mouse && cfg.mode.uses_mouse();
        pad::set_host_mouse_enabled(h.guest_mouse);
        h.publish();
    } else {
        return;
    }
    pad::configure_controller(cfg.features.speaker_volume, cfg.features.vibration_intensity, cfg.features.lightbar_color);
    log::info!("input: settings changed (device mode {})", cfg.mode.as_str());
    set_active(&cfg);
    *PENDING.lock().unwrap_or_else(|e| e.into_inner()) = Some(cfg);
    if let Some(on) = capture {
        request_window(WindowRequest::CaptureMouse(on));
    }
}

/// The menu opened or closed: release what the game holds (and the mouse)
/// or give mouse-look back.
fn menu_changed(open: bool) {
    let mut capture = None;
    if let Some(mut h) = hub() {
        if open {
            h.menu_look = h.kbm.mouse_look;
            h.menu_touch = h.kbm.touch_mode;
            capture = h.kbm.release_all().capture_mouse;
            if h.guest_mouse && h.mouse_buttons != 0 {
                h.mouse_buttons = 0;
                pad::push_host_mouse_event(0, 0, 0, 0, 0);
            }
        } else {
            let (look, touch) = (std::mem::take(&mut h.menu_look), std::mem::take(&mut h.menu_touch));
            if touch {
                h.kbm.set_touch_mode(true);
            }
            if look {
                h.kbm.set_mouse_look(true);
            }
            if (look || touch) && h.mode.uses_mouse() {
                capture = Some(true);
            }
        }
        h.publish();
    }
    menu::clear();
    if let Some(on) = capture {
        request_window(WindowRequest::CaptureMouse(on));
    }
}

fn resolve_file_settings() -> Config {
    let title = ps5hle::app_info::title_id();
    match profile::resolve(title.as_deref()) {
        Ok(r) => {
            if let Some(p) = &r.global {
                log::info!("input: settings from {}", p.display());
            }
            if let Some(p) = &r.profile {
                log::info!("input: per-game profile {}", p.display());
            }
            if let Some(e) = &r.profile_error {
                log::error!("input: {e}; using the global settings");
            }
            r.config
        }
        Err(e) => {
            log::error!("input: {e}; using the built-in mapping");
            Config::default()
        }
    }
}

/// Controller snapshot settings from the configuration.
pub fn pad_config(cfg: &Config) -> sdl::PadConfig {
    sdl::PadConfig {
        bindings: cfg.pad_bindings.clone(),
        left_stick: cfg.left_stick,
        right_stick: cfg.right_stick,
        triggers: cfg.triggers,
        motion: cfg.features.motion,
        touchpad: cfg.features.touchpad,
    }
}

pub fn output_features(cfg: &Config) -> sdl::OutputFeatures {
    let f = &cfg.features;
    sdl::OutputFeatures {
        rumble: f.rumble,
        rumble_strength: f.rumble_strength,
        lightbar: f.lightbar,
        adaptive_triggers: f.adaptive_triggers,
        lightbar_color: f.lightbar_color,
    }
}

/// Whether `RPS5E_NO_HOST_INPUT` asks for a neutral pad.
pub fn host_input_disabled() -> bool {
    std::env::var("RPS5E_NO_HOST_INPUT").is_ok_and(|v| !v.is_empty() && v != "0")
}

/// Start host input (idempotent). Never fails: a bad settings file falls
/// back to the built-in mapping, a missing SDL3 disables controllers.
pub fn start() {
    static STARTED: OnceLock<()> = OnceLock::new();
    if STARTED.set(()).is_err() {
        return;
    }
    start_script();
    start_replay();
    // Unattended runs: keep the pad neutral whatever reaches the window.
    if host_input_disabled() {
        log::info!("input: host input disabled (RPS5E_NO_HOST_INPUT)");
        return;
    }
    let cfg = resolve_settings();
    let players = cfg.players.clamp(1, MAX_PLAYERS);
    ps5hle::user_service::set_local_user_count(players);
    log::info!("input: device mode {}, {} local player(s)", cfg.mode.as_str(), players);
    let guest_mouse = cfg.guest_mouse && cfg.mode.uses_mouse();
    pad::set_host_mouse_enabled(guest_mouse);
    let kbm = KbmState::from_config(&cfg);
    set_active(&Config { players, ..cfg.clone() });
    if let Some(text) = startup_hint(&kbm, cfg.mode) {
        show_hint(text, HINT_TIME);
    }
    let _ = HUB.set(Mutex::new(Hub {
        kbm,
        mode: cfg.mode,
        players,
        pads: [None; MAX_PLAYERS],
        guest_mouse,
        mouse_buttons: 0,
        menu_look: false,
        menu_touch: false,
    }));
    ps5gpu::set_input_sink(Some(Box::new(on_event)));
    ps5gpu::menu::on_open_change(menu_changed);
    let shared = output::SharedOutput::default();
    pad::set_host_pad_output(Some(Box::new(output::OutputSink(shared.clone()))));
    // Pad speaker / vibration ports reach a DualSense once SDL is up.
    if cfg.gamepad {
        ps5hle::pad_audio::set_pad_audio_opener(Some(haptics::open_port));
    }
    pad::configure_controller(cfg.features.speaker_volume, cfg.features.vibration_intensity, cfg.features.lightbar_color);

    let (done_tx, done_rx) = std::sync::mpsc::channel::<()>();
    let spawned = std::thread::Builder::new().name("rps5e-input".into()).spawn(move || {
        input_thread(cfg, shared);
        let _ = done_tx.send(());
    });
    match spawned {
        // Process shutdown: the thread turns controller output off and
        // closes the controllers.
        Ok(_) => ps5vm::shutdown::on_shutdown(ps5vm::shutdown::ORDER_INPUT, "input", move || {
            let _ = done_rx.recv_timeout(INPUT_STOP_WAIT);
        }),
        Err(e) => log::warn!("input: cannot spawn the input thread: {e}"),
    }
}

/// How long shutdown waits for the input thread to release the controllers.
const INPUT_STOP_WAIT: Duration = Duration::from_millis(400);

/// One scripted input: buttons, or the left/right stick pushed fully in a
/// direction (`lup`, `ldown`, `lleft`, `lright`, `rup`, ...).
fn script_state(name: &str) -> Option<HostPadState> {
    let mut st = HostPadState::NEUTRAL;
    let n = name.to_ascii_lowercase();
    let stick = |dir: &str| -> Option<(u8, u8)> {
        Some(match dir {
            "up" => (0x80, 0x00),
            "down" => (0x80, 0xFF),
            "left" => (0x00, 0x80),
            "right" => (0xFF, 0x80),
            _ => return None,
        })
    };
    if let Some((x, y)) = n.strip_prefix('l').and_then(stick) {
        (st.left_x, st.left_y) = (x, y);
    } else if let Some((x, y)) = n.strip_prefix('r').and_then(stick) {
        (st.right_x, st.right_y) = (x, y);
    } else {
        // `a+b` presses several buttons together.
        for b in n.split('+') {
            st.buttons |= script_button(b)?;
        }
    }
    Some(st)
}

fn script_button(name: &str) -> Option<u32> {
    use pad::button::*;
    Some(match name.to_ascii_lowercase().as_str() {
        "cross" | "x" => CROSS,
        "circle" | "o" => CIRCLE,
        "square" => SQUARE,
        "triangle" => TRIANGLE,
        "options" | "start" => OPTIONS,
        "up" => UP,
        "down" => DOWN,
        "left" => LEFT,
        "right" => RIGHT,
        "l1" => L1,
        "r1" => R1,
        "l2" => L2,
        "r2" => R2,
        "touch" => TOUCH_PAD,
        _ => return None,
    })
}

/// Scripted button taps (`RPS5E_PAD_SCRIPT`).
fn start_script() {
    let Ok(script) = std::env::var("RPS5E_PAD_SCRIPT") else { return };
    // Entry: `<seconds>:<input>[:<hold ms>]` (hold defaults to 150 ms); the
    // input is a button, `a+b` buttons together, or a stick direction.
    let mut taps: Vec<(f64, HostPadState, u64)> = Vec::new();
    for item in script.split(',') {
        let mut parts = item.trim().split(':');
        let (Some(t), Some(b)) = (parts.next(), parts.next()) else { continue };
        let hold = parts.next().map(|h| h.trim().parse::<u64>().ok());
        match (t.trim().parse::<f64>(), script_state(b.trim()), hold) {
            (Ok(t), Some(b), None) => taps.push((t, b, 150)),
            (Ok(t), Some(b), Some(Some(h))) => taps.push((t, b, h)),
            _ => log::warn!("input: bad pad script entry '{item}'"),
        }
    }
    taps.sort_by(|a, b| a.0.total_cmp(&b.0));
    log::info!("input: pad script with {} taps", taps.len());
    let start = Instant::now();
    let _ = std::thread::Builder::new().name("rps5e-pad-script".into()).spawn(move || {
        for (t, b, hold) in taps {
            let at = start + Duration::from_secs_f64(t);
            let now = Instant::now();
            if at > now {
                std::thread::sleep(at - now);
            }
            log::info!("input: script tap {:#x} sticks ({},{}) ({},{}) at {t}s", b.buttons, b.left_x, b.left_y, b.right_x, b.right_y);
            pad::set_host_pad_state(b);
            std::thread::sleep(Duration::from_millis(hold));
            pad::set_host_pad_state(HostPadState::NEUTRAL);
        }
    });
}

/// `RPS5E_INPUT_REPLAY=<file>`: feed a recording made with
/// `RPS5E_INPUT_RECORD` to the primary pad at its recorded times.
fn start_replay() {
    let Some(path) = std::env::var_os("RPS5E_INPUT_REPLAY").filter(|p| !p.is_empty()) else { return };
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => {
            log::warn!("input: cannot read the recording {}: {e}", path.to_string_lossy());
            return;
        }
    };
    let events: Vec<(f64, HostPadState)> = text.lines().filter_map(pad::parse_input_line).collect();
    log::info!("input: replaying {} pad changes from {} ({:.0} s)", events.len(), path.to_string_lossy(), events.last().map_or(0.0, |e| e.0));
    let _ = std::thread::Builder::new().name("rps5e-input-replay".into()).spawn(move || {
        for (t, st) in events {
            while !ps5vm::shutdown::requested() {
                let left = t - pad::input_clock_secs();
                if left <= 0.0 {
                    break;
                }
                std::thread::sleep(Duration::from_secs_f64(left.min(0.05)));
            }
            if ps5vm::shutdown::requested() {
                return;
            }
            pad::set_host_pad_state(st);
        }
        log::info!("input: recording replay finished");
    });
}

fn controller_selectors(cfg: &Config) -> Vec<(usize, String)> {
    cfg.controller_players().into_iter().map(|(p, s)| (p, s.to_string())).collect()
}

fn input_thread(cfg: Config, shared: output::SharedOutput) {
    let mut controller_players = controller_selectors(&cfg);
    let mut sdl: Option<sdl::SdlGamepads> = None;
    let mut sdl_tried = false;
    let mut gamepad = cfg.gamepad;
    let mut pad_cfg = pad_config(&cfg);
    let mut features = output_features(&cfg);
    let mut assigned: Vec<Option<u32>> = vec![None; MAX_PLAYERS];
    let mut first = true;
    let mut nav = menu::NavState::default();
    let mut menu_was_open = false;
    let mut menu_button_held = false;
    while !ps5vm::shutdown::requested() {
        // Settings edited in the menu.
        if let Some(c) = PENDING.lock().unwrap_or_else(|e| e.into_inner()).take() {
            controller_players = controller_selectors(&c);
            pad_cfg = pad_config(&c);
            features = output_features(&c);
            gamepad = c.gamepad;
            first = true;
            if !gamepad {
                if let Some(s) = sdl.as_mut() {
                    s.release_all();
                }
            }
        }
        // Controllers are opened once they are wanted (at start, or when
        // the menu turns them on).
        if sdl.is_none() && !sdl_tried && gamepad && !controller_players.is_empty() {
            sdl_tried = true;
            match sdl::SdlGamepads::new() {
                Ok(s) => {
                    haptics::start();
                    sdl = Some(s);
                    menu::set_error(None);
                }
                Err(e) => {
                    log::info!("input: {e}; game controllers disabled");
                    menu::set_error(Some(e));
                }
            }
        }
        let open = ps5gpu::menu::is_open();
        let mut snaps: [Option<HostPadState>; MAX_PLAYERS] = [None; MAX_PLAYERS];
        if let Some(s) = sdl.as_mut().filter(|_| gamepad) {
            if s.pump() || first {
                first = false;
                let sel: Vec<(usize, &str)> = controller_players.iter().map(|(p, s)| (*p, s.as_str())).collect();
                let next = assign::assign(&sel, &s.keys(), &assigned, MAX_PLAYERS);
                if next != assigned {
                    let devices = s.devices();
                    for (p, id) in next.iter().enumerate() {
                        if *id != assigned[p] {
                            match id.and_then(|id| devices.iter().find(|d| d.id == id)) {
                                Some(d) => log::info!("input: player {} -> '{}'", p + 1, d.name),
                                None if assigned[p].is_some() => log::info!("input: player {} has no controller", p + 1),
                                None => {}
                            }
                        }
                    }
                    for d in &devices {
                        let player = next.iter().position(|id| *id == Some(d.id));
                        s.set_player_index(d.id, player);
                    }
                    // A new controller starts without the old one's
                    // vibration, trigger effects and motion state.
                    for (p, id) in next.iter().enumerate() {
                        if *id != assigned[p] {
                            pad::host_controller_changed(p);
                        }
                    }
                    haptics::set_active_controller(next[0]);
                    assigned = next;
                }
            }
            let outs = *shared.lock().unwrap_or_else(|e| e.into_inner());
            for (p, id) in assigned.iter().enumerate() {
                if let Some(id) = *id {
                    s.apply_output(id, &outs[p], &features);
                    snaps[p] = s.snapshot(id, &pad_cfg);
                }
            }
            // The PS button opens / closes the menu (player 1's controller,
            // or any one when player 1 has none).
            if ps5gpu::menu::installed() && menu::menu_button_free(&pad_cfg.bindings) {
                let ids: Vec<u32> = match assigned[0] {
                    Some(id) => vec![id],
                    None => s.devices().iter().map(|d| d.id).collect(),
                };
                let held = ids.iter().any(|&id| s.pressed_inputs(id).contains(&menu::MENU_BUTTON));
                if held && !menu_button_held {
                    ps5gpu::menu::toggle();
                }
                menu_button_held = held;
            }
            if open {
                let devices = s.devices();
                let pressed = devices.iter().map(|d| (d.id, s.pressed_inputs(d.id))).collect();
                menu::publish(devices, pressed, &[]);
            }
        }
        if open {
            let now = Instant::now();
            let state = snaps.iter().flatten().next().copied().unwrap_or(HostPadState::NEUTRAL);
            let mut steps = Vec::new();
            if menu_was_open {
                nav.update(&state, now, &mut steps);
            } else {
                nav.reset(&state);
            }
            if !steps.is_empty() {
                let c = menu::controllers();
                menu::publish(c.devices, c.pressed, &steps);
                ps5gpu::input::wake_window();
            }
        }
        menu_was_open = open;
        if let Some(mut h) = hub() {
            h.pads = snaps;
            h.kbm.tick(Instant::now());
            h.publish();
        }
        std::thread::sleep(POLL_INTERVAL);
    }
    // Shutdown: no vibration, speaker or light bar left on.
    if let Some(h) = haptics::instance() {
        h.shutdown();
    }
    drop(sdl);
}
