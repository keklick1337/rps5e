// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The launcher's frame around the pages: the sidebar (logo, navigation,
//! the running program with Stop), the status bar (messages, Undo, shader
//! compile progress) and the About page.

use crate::app::{Action, App, Page, RunKind};
use crate::launcher;
use crate::theme::{self, pal};
use eframe::egui::{self, Color32, CornerRadius, FontId, RichText, Sense, Vec2};

/// Project links.
pub const REPOSITORY_URL: &str = "https://github.com/keklick1337/rps5e";
pub const ISSUES_URL: &str = "https://github.com/keklick1337/rps5e/issues";

impl App {
    pub(crate) fn sidebar(&mut self, ui: &mut egui::Ui, actions: &mut Vec<Action>) {
        let p = pal(ui);
        ui.horizontal(|ui| {
            if let Some(t) = theme::logo(ui.ctx()) {
                ui.add(egui::Image::from_texture(egui::load::SizedTexture::new(t.id(), Vec2::splat(40.0))));
            }
            ui.vertical(|ui| {
                ui.spacing_mut().item_spacing.y = 0.0;
                ui.label(RichText::new("RustPS5Emu").size(17.0).strong().color(p.text));
                ui.label(RichText::new(format!("v{}", env!("CARGO_PKG_VERSION"))).size(11.5).color(p.weak));
            });
        });
        ui.add_space(22.0);
        let (_, errors) = self.log.lock().unwrap_or_else(|e| e.into_inner()).problems();
        let items = [
            (Page::Library, "🎮", "Library", if self.games.is_empty() { String::new() } else { self.games.len().to_string() }),
            (Page::Settings, "⚙", "Settings", String::new()),
            (Page::Logs, "📋", "Logs", if errors > 0 { errors.to_string() } else { String::new() }),
            (Page::About, "ℹ", "About", String::new()),
        ];
        for (page, icon, label, badge) in items {
            if nav_item(ui, self.page == page, icon, label, &badge, page == Page::Logs && errors > 0) {
                actions.push(Action::Show(page));
            }
        }
        ui.with_layout(egui::Layout::bottom_up(egui::Align::Min), |ui| {
            ui.add_space(2.0);
            self.running_card(ui, actions);
        });
    }

    /// The running program (or how the last one ended), with Stop.
    fn running_card(&self, ui: &mut egui::Ui, actions: &mut Vec<Action>) {
        let p = pal(ui);
        let Some((r, kind)) = &self.running else { return };
        let live = r.exit.is_none();
        theme::card(ui).inner_margin(egui::Margin::same(12)).show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                theme::dot(
                    ui,
                    if live {
                        p.ok
                    } else if r.exit.is_some_and(|s| s.success()) || r.stopped {
                        p.weak
                    } else {
                        p.danger
                    },
                    8.0,
                );
                let head = if live { format!("{} running", kind.label()) } else { format!("{} ended", kind.label()) };
                ui.label(RichText::new(head.to_uppercase()).size(10.5).strong().color(p.weak));
            });
            ui.add(egui::Label::new(RichText::new(&r.title).strong().color(p.text)).truncate());
            let secs = r.ran_for.unwrap_or_else(|| r.started.elapsed()).as_secs();
            let line = match r.exit {
                None => format!("pid {} · {}", r.child.id(), theme::duration_text(secs)),
                Some(st) => format!("{} · {}", if r.stopped { "stopped".into() } else { launcher::describe_exit(&st) }, theme::duration_text(secs)),
            };
            ui.label(RichText::new(line).small().color(p.weak));
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                let w = (ui.available_width() - ui.spacing().item_spacing.x) / 2.0;
                if theme::button(ui, true, "Log", Vec2::new(w, 28.0)).on_hover_text("The live log").clicked() {
                    actions.push(Action::Show(Page::Logs));
                }
                if live {
                    if theme::danger_button(ui, "Stop", Vec2::new(w, 28.0)).on_hover_text("Kill the process").clicked() {
                        actions.push(Action::Stop);
                    }
                } else if let RunKind::Game(k) = kind {
                    let can = self.game(k).is_some_and(|g| g.can_run());
                    if theme::button(ui, can, "Play again", Vec2::new(w, 28.0)).clicked() {
                        actions.push(Action::Launch(k.clone()));
                    }
                }
            });
        });
    }

    pub(crate) fn status_bar(&mut self, ui: &mut egui::Ui, actions: &mut Vec<Action>) {
        let p = pal(ui);
        ui.horizontal_centered(|ui| {
            if let Some((frac, line)) = self.recompile_progress() {
                ui.add(egui::ProgressBar::new(frac).text(line).desired_width(420.0).desired_height(18.0));
            } else {
                ui.add(egui::Label::new(RichText::new(&self.status).color(p.weak)).truncate());
            }
            if let Some((what, _)) = &self.undo {
                if ui.small_button("Undo").on_hover_text(format!("Undo: {what}")).clicked() {
                    actions.push(Action::Undo);
                }
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let roots = self.settings.library_roots().len();
                let off = self.settings.folder_rows().iter().filter(|r| !r.enabled).count();
                let mut text = format!(
                    "{} game{} · {roots} folder{}",
                    self.games.len(),
                    if self.games.len() == 1 { "" } else { "s" },
                    if roots == 1 { "" } else { "s" }
                );
                if off > 0 {
                    text.push_str(&format!(" ({off} off)"));
                }
                ui.label(RichText::new(text).small().color(p.weak));
            });
        });
    }

    pub(crate) fn about_page(&mut self, ui: &mut egui::Ui, actions: &mut Vec<Action>) {
        let p = pal(ui);
        egui::ScrollArea::vertical().auto_shrink([false; 2]).show(ui, |ui| {
            ui.add_space(16.0);
            ui.vertical_centered(|ui| {
                if let Some(t) = theme::logo(ui.ctx()) {
                    ui.add(egui::Image::from_texture(egui::load::SizedTexture::new(t.id(), Vec2::splat(128.0))));
                }
                ui.add_space(8.0);
                ui.label(RichText::new("RustPS5Emu").size(30.0).strong().color(p.text));
                ui.label(
                    RichText::new(format!("RPS5E {} · {} {}", env!("CARGO_PKG_VERSION"), std::env::consts::OS, std::env::consts::ARCH)).color(p.weak),
                );
                ui.add_space(6.0);
                ui.label(
                    RichText::new(
                        "A PlayStation 5 compatibility layer: games run natively, system libraries are reimplemented, graphics go through Vulkan.",
                    )
                    .color(p.text),
                );
                ui.label(RichText::new("Free software under the GNU General Public License, version 2 only.").small().color(p.weak));
                ui.add_space(14.0);
                ui.horizontal(|ui| {
                    let w = 180.0;
                    ui.add_space(((ui.available_width() - 3.0 * w - 2.0 * ui.spacing().item_spacing.x) / 2.0).max(0.0));
                    if theme::primary_button(ui, true, "Source code", Vec2::new(w, 34.0)).on_hover_text(REPOSITORY_URL).clicked() {
                        self.open_url(REPOSITORY_URL);
                    }
                    if theme::button(ui, true, "Report a bug", Vec2::new(w, 34.0)).on_hover_text(ISSUES_URL).clicked() {
                        self.open_url(ISSUES_URL);
                    }
                    let feed = !self.settings.network.update_feed_url.trim().is_empty();
                    let checking = self.update_rx.is_some();
                    let r = theme::button(ui, feed && !checking, if checking { "Checking..." } else { "Check for updates" }, Vec2::new(w, 34.0))
                        .on_hover_text("Asks the update feed set in Settings > Interface")
                        .on_disabled_hover_text("No update feed set (Settings > Interface)");
                    if r.clicked() {
                        self.check_updates();
                    }
                });
            });
            ui.add_space(22.0);
            let data = self.settings.emulator.effective_data_dir();
            let folders = [
                ("Settings", self.config_dir.clone(), "Launcher settings, game settings, input profiles"),
                ("Data", Some(data.clone()), "Save data, installed games, debug output"),
                ("Logs", Some(launcher::logs_dir()), "One log per run, the newest 30 kept"),
                ("Emulator", self.settings.emulator_binary().parent().map(|d| d.to_path_buf()), "The program games run with"),
            ];
            let width = ui.available_width().min(760.0);
            let margin = ((ui.available_width() - width) / 2.0).max(0.0);
            ui.horizontal(|ui| {
                ui.add_space(margin);
                ui.vertical(|ui| {
                    ui.set_width(width);
                    theme::card(ui).show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        theme::card_title(ui, "Folders", None);
                        for (name, dir, what) in folders {
                            ui.horizontal(|ui| {
                                ui.add_sized([80.0, 20.0], egui::Label::new(RichText::new(name).strong()));
                                ui.vertical(|ui| {
                                    ui.spacing_mut().item_spacing.y = 1.0;
                                    ui.add(egui::Label::new(dir.as_ref().map(|d| d.display().to_string()).unwrap_or_else(|| "-".into())).truncate());
                                    ui.label(RichText::new(what).small().color(p.weak));
                                });
                                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                    if let Some(d) = dir {
                                        if ui.add_enabled(d.is_dir(), egui::Button::new("Open")).clicked() {
                                            actions.push(Action::OpenPath(d));
                                        }
                                    }
                                });
                            });
                            ui.add_space(4.0);
                        }
                    });
                    ui.add_space(10.0);
                    theme::card(ui).show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        theme::card_title(ui, "Shortcuts", None);
                        egui::Grid::new("about-keys").num_columns(2).spacing([14.0, 6.0]).show(ui, |ui| {
                            for (k, what) in [
                                ("Double-click", "Play the game"),
                                ("Right-click", "The game's menu"),
                                ("F5", "Scan the library folders again"),
                                ("F1 / PS button", "In game: the menu (settings, controls, exit)"),
                                ("F2 (hold)", "In game: the current controls over the game"),
                                ("F10", "In game: performance overlay"),
                                ("F11", "In game: fullscreen"),
                                ("F12 / F8", "In game: screenshot (Create) / sound off and on (mute)"),
                            ] {
                                ui.label(RichText::new(k).monospace().color(p.accent));
                                ui.label(what);
                                ui.end_row();
                            }
                        });
                    });
                });
            });
        });
    }

    fn open_url(&mut self, url: &str) {
        if let Err(e) = crate::net::open_external(url) {
            self.status = format!("Could not open the browser: {e}");
        }
    }
}

/// One sidebar entry; true when clicked.
fn nav_item(ui: &mut egui::Ui, selected: bool, icon: &str, label: &str, badge: &str, alert: bool) -> bool {
    let p = pal(ui);
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 38.0), Sense::click());
    let bg = if selected {
        p.accent
    } else if resp.hovered() {
        p.hover
    } else {
        Color32::TRANSPARENT
    };
    ui.painter().rect_filled(rect, CornerRadius::same(8), bg);
    let fg = if selected { p.accent_text } else { p.text };
    ui.painter().text(rect.left_center() + Vec2::new(12.0, 0.0), egui::Align2::LEFT_CENTER, icon, FontId::proportional(16.0), fg);
    ui.painter().text(rect.left_center() + Vec2::new(40.0, 0.0), egui::Align2::LEFT_CENTER, label, FontId::proportional(15.0), fg);
    if !badge.is_empty() {
        let galley = ui.painter().layout_no_wrap(badge.to_string(), FontId::proportional(11.0), Color32::WHITE);
        let size = Vec2::new(galley.size().x + 12.0, 18.0);
        let r = egui::Rect::from_center_size(rect.right_center() - Vec2::new(size.x / 2.0 + 10.0, 0.0), size);
        let fill = if alert {
            p.danger
        } else if selected {
            Color32::from_white_alpha(60)
        } else {
            p.border
        };
        ui.painter().rect_filled(r, CornerRadius::same(9), fill);
        let text = if alert || selected { Color32::WHITE } else { p.weak };
        ui.painter().galley(r.center() - galley.size() / 2.0, galley, text);
    }
    ui.add_space(2.0);
    resp.clicked()
}
