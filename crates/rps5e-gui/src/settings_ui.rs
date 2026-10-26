// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The settings page: Library (library folders with their on/off switch
//! and game counts, single games, hidden games), Emulator (the global run
//! settings, also used by the game settings window, and the game settings
//! import/export), Input (devices, controller light bar and levels, key
//! bindings) and Interface (theme, emulator binary, network).

use crate::app::{Action, App, FilePurpose, Tab};
use crate::config::{self, EmulatorSettings};
use crate::input_store::Target;
use crate::input_ui;
use crate::sections::Section;
use crate::settings::{LibraryView, Theme};
use crate::theme::{self, pal};
use eframe::egui::{self, Color32, RichText, Sense, Stroke, Vec2};
use ps5input::mapping::{pad_button_actions, Config};
use ps5input::sdl::DeviceInfo;
use std::time::Duration;

/// Width of the settings column.
const COLUMN: f32 = 880.0;

impl App {
    pub(crate) fn settings_page(&mut self, ui: &mut egui::Ui) {
        theme::page_title(ui, "Settings", "");
        ui.add_space(6.0);
        let mut tab = self.tab;
        theme::segmented(ui, &mut tab, &[(Tab::Library, "Library"), (Tab::Emulator, "Emulator"), (Tab::Input, "Input and controls"), (Tab::Interface, "Interface")]);
        if tab != self.tab {
            self.bindings.capture = None;
            self.tab = tab;
        }
        ui.add_space(10.0);
        let mut actions = Vec::new();
        egui::ScrollArea::vertical().auto_shrink([false; 2]).show(ui, |ui| {
            ui.set_max_width(COLUMN);
            match self.tab {
                Tab::Library => self.library_tab(ui, &mut actions),
                Tab::Emulator => self.emulator_tab(ui),
                Tab::Input => theme::card(ui).show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    self.input_tab(ui)
                }).inner,
                Tab::Interface => self.interface_tab(ui),
            }
            ui.add_space(12.0);
        });
        for a in actions {
            self.do_action(a);
        }
    }

    fn library_tab(&mut self, ui: &mut egui::Ui, actions: &mut Vec<Action>) {
        let p = pal(ui);
        let counts = self.folder_counts();
        theme::card(ui).show(ui, |ui| {
            ui.set_width(ui.available_width());
            theme::card_title(
                ui,
                "Library folders",
                Some("Searched with their subfolders for game folders (eboot.bin and sce_sys/param.json) and runnable game packages (.pkg, .ffpkg). Turn a folder off to stop scanning it without removing it."),
            );
            let rows = self.settings.folder_rows();
            if rows.is_empty() {
                ui.label(RichText::new("No library folders yet.").color(p.weak));
            }
            for row in rows {
                egui::Frame::new().fill(ui.visuals().faint_bg_color).corner_radius(8).inner_margin(egui::Margin::symmetric(10, 8)).show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.horizontal(|ui| {
                        let mut on = row.enabled;
                        if theme::toggle(ui, &mut on).on_hover_text(if on { "Scanned: click to turn off" } else { "Off: click to scan it again" }).changed() {
                            actions.push(Action::SetFolderEnabled(row.path.clone(), on));
                        }
                        ui.vertical(|ui| {
                            ui.spacing_mut().item_spacing.y = 1.0;
                            let name = row.path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| row.path.display().to_string());
                            ui.horizontal(|ui| {
                                ui.label(RichText::new(name).strong().color(if row.enabled { p.text } else { p.weak }));
                                if !row.path.is_dir() {
                                    theme::pill(ui, "missing", p.danger);
                                } else if !row.enabled {
                                    theme::pill(ui, "off", p.weak);
                                } else {
                                    let n = counts.get(&row.path).copied().unwrap_or(0);
                                    let text = if self.scan_rx.is_some() { "scanning".to_string() } else { format!("{n} game{}", if n == 1 { "" } else { "s" }) };
                                    theme::pill(ui, &text, p.accent);
                                }
                                if row.from_env {
                                    theme::pill(ui, "$GAMES_DIR", p.weak).on_hover_text("Named by the GAMES_DIR environment variable");
                                }
                            });
                            ui.add(egui::Label::new(RichText::new(row.path.display().to_string()).small().color(p.weak)).truncate());
                        });
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if !row.from_env && ui.button("Remove...").on_hover_text("Take it out of the list (nothing is deleted)").clicked() {
                                actions.push(Action::RemoveSection(Section::Folder(row.path.clone())));
                            }
                            if ui.add_enabled(row.path.is_dir(), egui::Button::new("Open")).clicked() {
                                actions.push(Action::OpenPath(row.path.clone()));
                            }
                        });
                    });
                });
            }
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                let r = ui.add(egui::TextEdit::singleline(&mut self.new_dir).hint_text("/path/to/games").desired_width(320.0));
                let enter = r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                if (ui.add_enabled(!self.new_dir.trim().is_empty(), egui::Button::new("Add")).clicked() || enter) && !self.new_dir.trim().is_empty() {
                    let p = config::expand_home(self.new_dir.trim());
                    self.new_dir.clear();
                    self.add_library_dir(p);
                }
                if ui.button("Browse...").clicked() {
                    actions.push(Action::Pick(FilePurpose::AddLibraryFolder));
                }
                if ui.button("Rescan").on_hover_text("Scan the enabled folders again (F5)").clicked() {
                    actions.push(Action::Rescan);
                }
            });
        });
        ui.add_space(10.0);
        theme::card(ui).show(ui, |ui| {
            ui.set_width(ui.available_width());
            theme::card_title(ui, "Single games", Some("Game folders, packages and executables added one by one."));
            let mut remove = None;
            for (i, d) in self.settings.extra_games.iter().enumerate() {
                ui.horizontal(|ui| {
                    if ui.small_button("Remove").on_hover_text("Take it out of the library (nothing is deleted)").clicked() {
                        remove = Some(i);
                    }
                    let missing = if d.exists() { "" } else { "  (missing)" };
                    ui.label(format!("{}{missing}", d.display()));
                });
            }
            if self.settings.extra_games.is_empty() {
                ui.label(RichText::new("None.").color(p.weak));
            }
            if let Some(i) = remove {
                self.settings.extra_games.remove(i);
                self.save_settings();
                self.rescan();
            }
            ui.horizontal(|ui| {
                if ui.button("Add game file...").on_hover_text("eboot.bin, .pkg / .ffpkg or .elf / .self").clicked() {
                    actions.push(Action::Pick(FilePurpose::AddGame));
                }
                if ui.button("Add game folder...").clicked() {
                    actions.push(Action::Pick(FilePurpose::AddGameFolder));
                }
            });
        });
        ui.add_space(10.0);
        theme::card(ui).show(ui, |ui| {
            ui.set_width(ui.available_width());
            theme::card_title(ui, "Hidden games", Some("Games removed from the library (game menu > Remove from library)."));
            if self.settings.hidden_games.is_empty() {
                ui.label(RichText::new("None.").color(p.weak));
                return;
            }
            let mut restore = None;
            for (i, h) in self.settings.hidden_games.iter().enumerate() {
                ui.horizontal(|ui| {
                    if ui.small_button("Restore").clicked() {
                        restore = Some(i);
                    }
                    let id = if h.title_id.is_empty() { "-" } else { h.title_id.as_str() };
                    ui.label(format!("{}  {id}", h.title));
                    ui.add(egui::Label::new(RichText::new(h.path.display().to_string()).small().color(p.weak)).truncate());
                });
            }
            let all = ui.button("Restore all").clicked();
            if all {
                self.settings.hidden_games.clear();
            } else if let Some(i) = restore {
                self.settings.unhide(i);
            }
            if all || restore.is_some() {
                self.save_settings();
                self.rescan();
            }
        });
    }

    fn emulator_tab(&mut self, ui: &mut egui::Ui) {
        theme::card(ui).show(ui, |ui| {
            ui.set_width(ui.available_width());
            theme::card_title(ui, "Global settings", Some("Every game without its own settings runs with these (a game's own: its details > Settings...)."));
            let mut s = self.settings.emulator.clone();
            emulator_form(ui, &mut s, "global", false);
            if s != self.settings.emulator {
                self.settings.emulator = s;
                self.save_settings();
            }
            ui.add_space(6.0);
            if ui.button("Reset to defaults").clicked() {
                self.settings.emulator = EmulatorSettings::default();
                self.save_settings();
            }
        });
        ui.add_space(10.0);
        theme::card(ui).show(ui, |ui| {
            ui.set_width(ui.available_width());
            theme::card_title(ui, "Game settings files", Some("Share a game's own settings as JSON keyed by its PPSA code."));
            ui.horizontal(|ui| {
                if ui.button("Import...").clicked() {
                    self.open_file_dialog(FilePurpose::ImportConfigs);
                }
                let exportable: Vec<(String, String)> = self
                    .configs
                    .games
                    .iter()
                    .filter(|(_, g)| config::is_game_settings_title_id(&g.title_id.to_ascii_uppercase()))
                    .map(|(k, g)| (k.clone(), format!("{} - {}", g.title_id, g.name)))
                    .collect();
                ui.add_enabled_ui(!exportable.is_empty(), |ui| {
                    ui.menu_button("Export...", |ui| {
                        for (k, label) in exportable {
                            if ui.button(label).clicked() {
                                self.open_file_dialog(FilePurpose::ExportConfig(k));
                                ui.close_menu();
                            }
                        }
                    });
                });
                ui.label(RichText::new(format!("{} game(s) with their own settings", self.configs.games.len())).small().color(pal(ui).weak));
            });
        });
    }

    fn interface_tab(&mut self, ui: &mut egui::Ui) {
        let mut clear_shaders = false;
        let before = self.settings.clone();
        let p = pal(ui);
        theme::card(ui).show(ui, |ui| {
            ui.set_width(ui.available_width());
            theme::card_title(ui, "Appearance", None);
            egui::Grid::new("interface").num_columns(2).spacing([16.0, 10.0]).show(ui, |ui| {
                ui.label("Theme");
                theme::segmented(ui, &mut self.settings.theme, &[(Theme::System, "System"), (Theme::Dark, "Dark"), (Theme::Light, "Light")]);
                ui.end_row();
                ui.label("Library view");
                theme::segmented(ui, &mut self.settings.view, &[(LibraryView::Grid, "Grid"), (LibraryView::List, "List")]);
                ui.end_row();
            });
        });
        ui.add_space(10.0);
        theme::card(ui).show(ui, |ui| {
            ui.set_width(ui.available_width());
            theme::card_title(ui, "Emulator", None);
            egui::Grid::new("emu-binary").num_columns(2).spacing([16.0, 10.0]).show(ui, |ui| {
                ui.label("Emulator binary");
                ui.vertical(|ui| {
                    ui.add(egui::TextEdit::singleline(&mut self.settings.emulator_path).hint_text("empty = this program / rps5e-cli next to it").desired_width(420.0));
                    let runs = self.settings.emulator_binary();
                    let found = runs.is_file() || runs.components().count() == 1;
                    ui.label(RichText::new(format!("Runs: {}", runs.display())).small().color(if found { p.weak } else { p.danger }));
                });
                ui.end_row();
                ui.label("Package install");
                ui.label(RichText::new(match self.pkg_install {
                    Some(true) => "supported (install-pkg)",
                    Some(false) => "not supported by this emulator binary",
                    None => "checking...",
                }).color(p.weak));
                ui.end_row();
                ui.label("Settings folder");
                ui.label(self.config_dir.as_ref().map(|d| d.display().to_string()).unwrap_or_else(|| "-".into())).on_hover_text("Set with RPS5E_CONFIG_DIR");
                ui.end_row();
                ui.label("Shader caches");
                if ui
                    .add_enabled(!self.is_running(), egui::Button::new("Clear all shader caches..."))
                    .on_hover_text("Delete the compiled shaders and pipelines of every game; each game builds them again before its next start")
                    .clicked()
                {
                    clear_shaders = true;
                }
                ui.end_row();
            });
        });
        ui.add_space(10.0);
        theme::card(ui).show(ui, |ui| {
            ui.set_width(ui.available_width());
            theme::card_title(ui, "Network", Some("The launcher makes no network requests unless they are turned on here."));
            let n = &mut self.settings.network;
            ui.checkbox(&mut n.online_cheats, "Browse the online cheat collection in the cheats window");
            ui.horizontal(|ui| {
                ui.label("Update feed");
                ui.add(egui::TextEdit::singleline(&mut n.update_feed_url).hint_text("https://... (JSON: tag, html_url)").desired_width(420.0));
            });
            let has_feed = !n.update_feed_url.trim().is_empty();
            ui.add_enabled(has_feed, egui::Checkbox::new(&mut n.check_updates_on_startup, "Check for updates on startup"));
            ui.horizontal(|ui| {
                let checking = self.update_rx.is_some();
                if ui.add_enabled(has_feed && !checking, egui::Button::new("Check for updates now")).clicked() {
                    self.check_updates();
                }
                ui.label(RichText::new(format!("This build: {}", crate::updates::current_tag())).small().color(p.weak));
            });
        });
        if self.settings != before {
            self.save_settings();
            if self.settings.emulator_path != before.emulator_path {
                // Asked again (poll_background) with the new binary.
                self.pkg_install = None;
            }
        }
        if clear_shaders {
            self.do_action(Action::ClearAllShaderCaches);
        }
    }

    fn input_tab(&mut self, ui: &mut egui::Ui) {
        // Target: global settings or a game's profile.
        let current = self.input.as_ref().map(|s| s.target.clone()).unwrap_or(Target::Global);
        let mut next = current.clone();
        ui.horizontal(|ui| {
            ui.label("Settings for");
            let label = match &current {
                Target::Global => "All games (global)".to_string(),
                Target::Game { title, title_id } => format!("{title} ({title_id})"),
            };
            egui::ComboBox::from_id_salt("target").selected_text(label).width(320.0).show_ui(ui, |ui| {
                ui.selectable_value(&mut next, Target::Global, "All games (global)");
                for g in &self.games {
                    if let Some(id) = &g.title_id {
                        let t = Target::Game { title_id: id.clone(), title: g.title.clone() };
                        ui.selectable_value(&mut next, t, format!("{} ({id})", g.title));
                    }
                }
            });
        });
        if next != current || self.input.is_none() {
            self.load_input(next);
        }
        if let Some(e) = &self.input_error {
            ui.colored_label(Color32::from_rgb(255, 110, 110), e);
        }
        let Some(mut store) = self.input.take() else { return };
        if let Target::Game { .. } = store.target {
            let mut on = store.profile_enabled;
            if ui.checkbox(&mut on, "Use a per-game profile for this game").changed() {
                store.set_profile_enabled(on);
            }
            if !on {
                ui.label(RichText::new("This game uses the global settings.").italics());
            }
        }
        ui.add_enabled_ui(store.editable(), |ui| self.input_editor(ui, &mut store));
        ui.separator();
        let mut revert = false;
        ui.horizontal(|ui| {
            if ui.add_enabled(store.dirty, egui::Button::new("Save")).clicked() {
                match store.save() {
                    Ok(()) => self.status = format!("Saved {}", store.path().display()),
                    Err(e) => self.input_error = Some(e),
                }
            }
            if ui.add_enabled(store.dirty, egui::Button::new("Revert")).clicked() {
                revert = true;
            }
            if ui.add_enabled(store.editable(), egui::Button::new("Reset to defaults")).clicked() {
                store.config = Config::default();
                store.dirty = true;
            }
            ui.label(RichText::new(store.path().display().to_string()).small());
            if store.dirty {
                ui.label(RichText::new("unsaved").color(Color32::YELLOW));
            }
        });
        if revert {
            self.load_input(store.target.clone());
        } else if self.input.is_none() {
            self.input = Some(store);
        }
    }

    fn input_editor(&mut self, ui: &mut egui::Ui, store: &mut crate::input_store::InputStore) {
        let before = store.config.clone();
        let cfg = &mut store.config;
        input_ui::device_settings(ui, cfg, &self.devices, self.sdl_error.as_deref(), true);
        egui::CollapsingHeader::new("Controller: light bar, speaker, vibration")
            .default_open(true)
            .show(ui, |ui| input_ui::controller_settings(ui, cfg));
        ui.collapsing("Controller preview", |ui| self.controller_preview(ui, cfg));
        ui.collapsing("Analog and mouse", |ui| input_ui::analog_settings(ui, cfg));
        egui::CollapsingHeader::new("Key bindings").default_open(true).show(ui, |ui| self.bindings.show(ui, cfg));
        if store.config != before {
            store.dirty = true;
        }
    }

    fn controller_preview(&mut self, ui: &mut egui::Ui, cfg: &Config) {
        if self.devices.is_empty() {
            ui.label("No controller connected.");
            return;
        }
        let label = |d: &DeviceInfo| format!("{} ({})", d.name, d.kind.label());
        let current = self.devices.iter().find(|d| Some(d.id) == self.preview_device).map(label).unwrap_or_default();
        egui::ComboBox::from_id_salt("preview").selected_text(current).show_ui(ui, |ui| {
            for d in &self.devices {
                ui.selectable_value(&mut self.preview_device, Some(d.id), label(d));
            }
        });
        let (Some(id), Some(sdl)) = (self.preview_device, self.sdl.as_ref()) else { return };
        let Some(dev) = self.devices.iter().find(|d| d.id == id) else { return };
        let Some(s) = sdl.snapshot(id, &ps5input::pad_config(cfg)) else { return };
        ui.horizontal(|ui| {
            let mut info = vec![dev.kind.label().to_string()];
            if !dev.serial.is_empty() {
                info.push(format!("serial {}", dev.serial));
            }
            if let Some(b) = s.battery {
                info.push(if b.wireless { "wireless".into() } else { "wired".into() });
                if let Some(p) = b.percent {
                    info.push(format!("battery {p}%{}", if b.charging { " (charging)" } else { "" }));
                } else if b.charging {
                    info.push("charging".into());
                }
            }
            info.push(format!("motion {}", if dev.motion { "yes" } else { "no" }));
            info.push(format!("touch pad {}", if dev.touchpad { "yes" } else { "no" }));
            ui.label(info.join(" | "));
        });
        ui.horizontal(|ui| {
            stick_widget(ui, "L", s.left_x, s.left_y);
            stick_widget(ui, "R", s.right_x, s.right_y);
            ui.vertical(|ui| {
                ui.add(egui::ProgressBar::new(s.l2 as f32 / 255.0).text(format!("L2 {}", s.l2)).desired_width(140.0));
                ui.add(egui::ProgressBar::new(s.r2 as f32 / 255.0).text(format!("R2 {}", s.r2)).desired_width(140.0));
                touch_widget(ui, &s);
            });
        });
        ui.horizontal_wrapped(|ui| {
            for (name, bit) in pad_button_actions() {
                let on = s.buttons & bit != 0;
                let text = RichText::new(name).color(if on { Color32::BLACK } else { ui.visuals().weak_text_color() });
                let text = if on { text.background_color(Color32::LIGHT_GREEN) } else { text };
                ui.label(text);
            }
        });
        if let Some(m) = s.motion {
            ui.label(
                RichText::new(format!(
                    "accel {:+.2} {:+.2} {:+.2} m/s2   gyro {:+.2} {:+.2} {:+.2} rad/s",
                    m.accel[0], m.accel[1], m.accel[2], m.angular_velocity[0], m.angular_velocity[1], m.angular_velocity[2]
                ))
                .monospace(),
            );
        }
        ui.ctx().request_repaint_after(Duration::from_millis(33));
    }

}

/// The run settings form. `game`: a game's own configuration (shows the
/// executable name).
pub(crate) fn emulator_form(ui: &mut egui::Ui, s: &mut EmulatorSettings, salt: &str, game: bool) {
    let path_row = |ui: &mut egui::Ui, label: &str, v: &mut String, hint: &str| {
        ui.label(label);
        ui.add(egui::TextEdit::singleline(v).hint_text(hint).desired_width(340.0));
        ui.end_row();
    };
    ui.strong("Graphics");
    egui::Grid::new((salt, "gfx")).num_columns(2).spacing([16.0, 8.0]).show(ui, |ui| {
        ui.label("Present mode");
        let current = config::PRESENT_MODES.iter().find(|m| m.0 == s.present_mode).map(|m| m.1).unwrap_or("?");
        egui::ComboBox::from_id_salt((salt, "present")).selected_text(current).show_ui(ui, |ui| {
            for (k, label) in config::PRESENT_MODES {
                ui.selectable_value(&mut s.present_mode, k.to_string(), label);
            }
        });
        ui.end_row();
        ui.label("Fullscreen");
        ui.checkbox(&mut s.fullscreen, "Open the game window fullscreen (F11 toggles in game)");
        ui.end_row();
        ui.label("HDR");
        ui.checkbox(&mut s.hdr, "Enable HDR display support")
            .on_hover_text("HDR10 output for games that support it. Requires HDR enabled on the host display; SDR otherwise.");
        ui.end_row();
        ui.label("Performance overlay");
        ui.checkbox(&mut s.perf_overlay, "Show FPS, frame times, CPU/GPU load (F10 toggles in game)");
        ui.end_row();
        ui.label("FPS counter");
        ui.checkbox(&mut s.fps_counter, "Frames per second in the corner");
        ui.end_row();
        for (label, value, options, id) in [
            ("Scaling", &mut s.scale_mode, &config::SCALE_MODES[..], "scale"),
            ("Scaling filter", &mut s.scale_filter, &config::SCALE_FILTERS[..], "filter"),
        ] {
            ui.label(label);
            let current = options.iter().find(|m| m.0 == value.as_str()).map(|m| m.1).unwrap_or("?");
            egui::ComboBox::from_id_salt((salt, id)).selected_text(current).show_ui(ui, |ui| {
                for (k, l) in options {
                    ui.selectable_value(value, k.to_string(), *l);
                }
            });
            ui.end_row();
        }
        ui.label("GPU fast paths");
        ui.checkbox(&mut s.fast_paths, "Per-draw fast paths (off: slower, for diagnosing glitches)");
        ui.end_row();
        ui.label("GPU");
        ui.add(egui::TextEdit::singleline(&mut s.gpu_device).hint_text("index or name, empty = automatic"));
        ui.end_row();
        ui.label("Refresh rate");
        ui.horizontal(|ui| {
            let mut custom = s.refresh_rate != 0.0;
            if ui.checkbox(&mut custom, "custom").changed() {
                s.refresh_rate = if custom { 60.0 } else { 0.0 };
            }
            if custom {
                ui.add(egui::DragValue::new(&mut s.refresh_rate).range(30.0..=360.0).speed(0.5).suffix(" Hz"));
            } else {
                ui.label("59.94 Hz");
            }
        });
        ui.end_row();
        ui.label("Headless");
        ui.checkbox(&mut s.headless, "No window (testing)");
        ui.end_row();
    });
    ui.strong("System");
    egui::Grid::new((salt, "sys")).num_columns(2).spacing([16.0, 8.0]).show(ui, |ui| {
        ui.label("Console language");
        let current = config::LANGUAGES.iter().find(|l| l.0 == s.console_language).map(|l| l.1).unwrap_or("?");
        egui::ComboBox::from_id_salt((salt, "lang")).selected_text(current).show_ui(ui, |ui| {
            for (id, name) in config::LANGUAGES {
                ui.selectable_value(&mut s.console_language, *id, *name);
            }
        });
        ui.end_row();
        ui.label("Skip notice screen");
        ui.checkbox(&mut s.skip_notice_screen, "Console setting games read");
        ui.end_row();
        ui.label("Audio");
        ui.checkbox(&mut s.mute_audio, "Mute (no audio device)");
        ui.end_row();
        ui.label("Volume");
        ui.add(egui::Slider::new(&mut s.volume, 0..=100).suffix(" %"));
        ui.end_row();
        ui.label("In-game menu");
        ui.checkbox(&mut s.pause_in_menu, "Pause the game while the menu (F1) is open");
        ui.end_row();
        ui.label("Input devices");
        let current = config::INPUT_MODES.iter().find(|m| m.0 == s.input_mode).map(|m| m.1).unwrap_or("?");
        egui::ComboBox::from_id_salt((salt, "inmode")).selected_text(current).width(240.0).show_ui(ui, |ui| {
            for (k, label) in config::INPUT_MODES {
                ui.selectable_value(&mut s.input_mode, k.to_string(), label);
            }
        });
        ui.end_row();
        ui.label("Host input");
        ui.checkbox(&mut s.disable_host_input, "Disable (neutral pad, for unattended runs)");
        ui.end_row();
        ui.label("Cheats");
        ui.checkbox(&mut s.cheats, "Apply the game's cheat file at start");
        ui.end_row();
        ui.label("Log level");
        egui::ComboBox::from_id_salt((salt, "log")).selected_text(s.log_level.clone()).show_ui(ui, |ui| {
            for l in config::LOG_LEVELS {
                ui.selectable_value(&mut s.log_level, l.to_string(), l);
            }
        });
        ui.end_row();
        if game {
            ui.label("Executable");
            ui.add(egui::TextEdit::singleline(&mut s.elf).hint_text(config::DEFAULT_ELF));
            ui.end_row();
        }
    });
    ui.strong("Folders");
    egui::Grid::new((salt, "paths")).num_columns(2).spacing([16.0, 8.0]).show(ui, |ui| {
        path_row(ui, "Data", &mut s.data_dir, &format!("default: {}", ps5hle::filesystem::user_data_dir().display()));
        path_row(ui, "Save data", &mut s.savedata_dir, "default: <data>/savedata");
        path_row(ui, "/temp0", &mut s.temp0_dir, "default: <data>/<title>/temp0 (emptied each start)");
        path_row(ui, "/download0", &mut s.download0_dir, "default: <data>/<title>/download0");
        path_row(ui, "System fonts", &mut s.system_fonts_dir, "default: host fonts");
    });
    egui::CollapsingHeader::new("GPU switches").id_salt((salt, "gpu-switches")).show(ui, |ui| {
        for sw in crate::ingame::GPU_SWITCHES {
            let mut on = sw.on_in(&s.extra_env);
            if ui.checkbox(&mut on, sw.label).on_hover_text(sw.env).changed() {
                sw.set_in(&mut s.extra_env, on);
            }
        }
    });
    ui.collapsing("Advanced", |ui| {
        egui::Grid::new((salt, "adv")).num_columns(2).spacing([16.0, 8.0]).show(ui, |ui| {
            path_row(ui, "Network redirect", &mut s.net_redirect, "address; empty = loopback only");
            path_row(ui, "Pipeline cache", &mut s.pipeline_cache, "file, or off; empty = default");
            ui.label("Environment");
            let mut text = s.extra_env.join("\n");
            if ui.add(egui::TextEdit::multiline(&mut text).hint_text("KEY=VALUE per line").desired_rows(3).desired_width(340.0)).changed() {
                s.extra_env = text.lines().map(str::trim).filter(|l| !l.is_empty()).map(str::to_string).collect();
            }
            ui.end_row();
        });
    });
    if let Err(e) = s.validate() {
        ui.colored_label(Color32::from_rgb(255, 110, 110), e);
    }
}

fn stick_widget(ui: &mut egui::Ui, name: &str, x: u8, y: u8) {
    let (rect, _) = ui.allocate_exact_size(Vec2::splat(84.0), Sense::hover());
    let p = ui.painter();
    let c = rect.center();
    let r = 38.0;
    p.circle_stroke(c, r, Stroke::new(1.0f32, ui.visuals().weak_text_color()));
    let pos = c + Vec2::new((x as f32 - 128.0) / 128.0 * r, (y as f32 - 128.0) / 128.0 * r);
    p.circle_filled(pos, 5.0, Color32::LIGHT_BLUE);
    p.text(rect.left_top(), egui::Align2::LEFT_TOP, name, egui::FontId::proportional(12.0), ui.visuals().text_color());
}

fn touch_widget(ui: &mut egui::Ui, s: &ps5input::HostPadState) {
    let (rect, _) = ui.allocate_exact_size(Vec2::new(140.0, 69.0), Sense::hover());
    let p = ui.painter();
    p.rect_stroke(rect, 4.0, Stroke::new(1.0f32, ui.visuals().weak_text_color()), egui::StrokeKind::Inside);
    for t in s.touch.iter().flatten() {
        let pos = rect.min + Vec2::new(t.x.clamp(0.0, 1.0) * rect.width(), t.y.clamp(0.0, 1.0) * rect.height());
        p.circle_filled(pos, 4.0, Color32::LIGHT_GREEN);
    }
}
