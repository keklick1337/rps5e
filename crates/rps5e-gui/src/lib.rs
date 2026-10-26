// RustPS5Emu (RPS5E)
// Copyright (C) 2025-2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! rps5e launcher (eframe/egui): game library (game folders and package
//! files, metadata, compatibility status), games run as child
//! `rps5e-cli run` processes (plain, or with debugging options) with a live
//! log page, global and per-game
//! configurations, key bindings, controller light bar, cheats, trophy
//! viewer, theme and an opt-in update check. Settings are kept in
//! `~/.config/rps5e/`.

pub mod app;
pub mod cheats;
pub mod compat;
pub mod config;
pub mod controls_overlay;
mod dialogs;
pub mod dualsense;
pub mod input_store;
pub mod ingame;
pub mod input_ui;
pub mod keys;
pub mod launcher;
pub mod library;
mod library_ui;
mod logs_ui;
mod pages;
pub mod run_options;
pub mod net;
pub mod report;
pub mod sections;
pub mod theme;
pub mod settings;
#[cfg(test)]
mod softrender;
mod settings_ui;
pub mod trophies;
pub mod updates;

/// Scratch directory of the unit tests (under the workspace `target/`).
#[cfg(test)]
pub(crate) fn test_dir() -> std::path::PathBuf {
    let d = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/rps5e-gui-test");
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Start-up options.
#[derive(Clone, Debug, Default)]
pub struct Options {
    /// Open the settings window at start.
    pub open_settings: bool,
}

/// Open the launcher window; returns when it is closed.
pub fn run() -> Result<(), String> {
    run_with(Options { open_settings: std::env::args().any(|a| a == "--settings") })
}

pub fn run_with(options: Options) -> Result<(), String> {
    let native = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title(ps5gpu::APP_TITLE)
            .with_app_id(ps5gpu::APP_ID)
            .with_icon(eframe::egui::IconData { rgba: ps5gpu::WINDOW_ICON_RGBA.to_vec(), width: ps5gpu::WINDOW_ICON_SIZE, height: ps5gpu::WINDOW_ICON_SIZE })
            .with_inner_size([1280.0, 800.0])
            .with_min_inner_size([960.0, 560.0]),
        ..Default::default()
    };
    // eframe panics when the driver offers no usable OpenGL configuration
    // (broken drivers, remote desktops): say so instead of crashing.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        eframe::run_native(ps5gpu::APP_ID, native, Box::new(move |cc| Ok(Box::new(app::App::new(cc, options)))))
    }));
    let error = match result {
        Ok(Ok(())) => return Ok(()),
        Ok(Err(e)) => e.to_string(),
        Err(_) => "the launcher window needs OpenGL 2.0 or newer, and the graphics driver offers no usable configuration. Update the graphics driver; games can still be started with `rps5e-cli run <game folder>`.".to_string(),
    };
    show_startup_error(&error);
    Err(error)
}

/// A message box on Windows, where a launcher started from Explorer has no
/// console to print to.
#[cfg(windows)]
fn show_startup_error(text: &str) {
    use windows_sys::Win32::UI::WindowsAndMessaging::{MessageBoxW, MB_ICONERROR, MB_OK};
    let wide = |s: &str| s.encode_utf16().chain(std::iter::once(0)).collect::<Vec<u16>>();
    let (text, title) = (wide(text), wide(ps5gpu::APP_TITLE));
    // SAFETY: both strings are NUL-terminated and outlive the call.
    unsafe { MessageBoxW(std::ptr::null_mut(), text.as_ptr(), title.as_ptr(), MB_OK | MB_ICONERROR) };
}

#[cfg(not(windows))]
fn show_startup_error(_text: &str) {}
