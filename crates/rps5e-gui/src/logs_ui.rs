// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The logs page (the running program with Stop and its command line, the
//! live log or a saved run log with a level filter and search, the run
//! log files) and the "Run with..." dialog of debugging options.

use crate::app::{Action, App, Page, RunDialog};
use crate::launcher::{self, LevelFilter, LineKind, Log};
use crate::run_options::{self, RunOptions};
use crate::theme::{self, pal};
use eframe::egui::{self, RichText, Vec2};
use std::time::Duration;

/// Run log files listed on the page.
const RECENT_LOGS: usize = 12;

impl App {
    pub(crate) fn logs_page(&mut self, ui: &mut egui::Ui, actions: &mut Vec<Action>) {
        let p = pal(ui);
        ui.horizontal(|ui| {
            theme::page_title(ui, "Logs", "output of the emulator runs");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("Open logs folder").on_hover_text(launcher::logs_dir().display().to_string()).clicked() {
                    actions.push(Action::OpenLogs);
                }
            });
        });
        ui.add_space(8.0);
        self.process_card(ui, actions);
        ui.add_space(10.0);
        egui::SidePanel::right("log-files")
            .exact_width(250.0)
            .resizable(false)
            .show_separator_line(false)
            .frame(egui::Frame::new().inner_margin(egui::Margin { left: 12, ..Default::default() }))
            .show_inside(ui, |ui| self.log_files(ui));
        egui::CentralPanel::default().frame(egui::Frame::new()).show_inside(ui, |ui| {
            // Toolbar: source, level, search.
            ui.horizontal(|ui| {
                if let Some((path, _)) = &self.log_view.file {
                    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                    theme::pill(ui, &name, p.warn).on_hover_text(path.display().to_string());
                    if ui.button("Back to live log").clicked() {
                        self.log_view.file = None;
                    }
                } else {
                    theme::pill(ui, "live", p.ok);
                }
                let mut level = self.log_view.level;
                let opts: Vec<(LevelFilter, &str)> = LevelFilter::ALL.iter().map(|l| (*l, l.label())).collect();
                if theme::segmented(ui, &mut level, &opts) {
                    self.log_view.level = level;
                }
                ui.add(egui::TextEdit::singleline(&mut self.log_view.query).hint_text("🔍 Search the log").desired_width(200.0));
                if !self.log_view.query.is_empty() && ui.small_button("✕").on_hover_text("Clear the search").clicked() {
                    self.log_view.query.clear();
                }
            });
            ui.add_space(4.0);
            let live = self.log.clone();
            let file = self.log_view.file.take();
            let mut clear = false;
            let copy = {
                let guard;
                let log: &Log = match &file {
                    Some((_, l)) => l,
                    None => {
                        guard = live.lock().unwrap_or_else(|e| e.into_inner());
                        &guard
                    }
                };
                self.log_lines(ui, log, file.is_none(), &mut clear)
            };
            if self.log_view.file.is_none() {
                self.log_view.file = file;
            }
            if let Some(t) = copy {
                self.copy_text = Some(t);
            }
            if clear {
                live.lock().unwrap_or_else(|e| e.into_inner()).clear();
            }
        });
        if self.is_running() {
            ui.ctx().request_repaint_after(Duration::from_millis(150));
        }
    }

    /// The counts, the line buttons and the lines of `log`; returns text
    /// to copy.
    fn log_lines(&mut self, ui: &mut egui::Ui, log: &Log, live: bool, clear: &mut bool) -> Option<String> {
        let p = pal(ui);
        {
            let shown = log.filtered(self.log_view.level, &self.log_view.query);
            let (w, e) = log.problems();
            let mut copy = None;
            ui.horizontal(|ui| {
                ui.label(RichText::new(format!("{} of {} lines", shown.len(), log.lines.len())).small().color(p.weak));
                if w > 0 {
                    theme::pill(ui, &format!("{w} warning{}", if w == 1 { "" } else { "s" }), p.warn);
                }
                if e > 0 {
                    theme::pill(ui, &format!("{e} error{}", if e == 1 { "" } else { "s" }), p.danger);
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if live && ui.button("Clear").on_hover_text("Empty the live log (the log file stays)").clicked() {
                        *clear = true;
                    }
                    if ui.button("Copy").on_hover_text("Copy the lines shown").clicked() {
                        let mut text = String::new();
                        for &i in &shown {
                            text.push_str(&log.lines[i].text);
                            text.push('\n');
                        }
                        copy = Some(text);
                    }
                    ui.checkbox(&mut self.log_view.follow, "Follow");
                });
            });
            let row_h = ui.text_style_height(&egui::TextStyle::Monospace) + 1.0;
            egui::Frame::new().fill(ui.visuals().extreme_bg_color).corner_radius(8).inner_margin(egui::Margin::same(8)).show(ui, |ui| {
                ui.set_min_size(ui.available_size());
                if log.lines.is_empty() {
                    ui.label(RichText::new("Nothing yet. Play a game and its output shows here.").color(p.weak));
                    return;
                }
                egui::ScrollArea::both().auto_shrink([false; 2]).stick_to_bottom(self.log_view.follow).show_rows(
                    ui,
                    row_h,
                    shown.len(),
                    |ui, range| {
                        for &i in &shown[range] {
                            let l = &log.lines[i];
                            let color = match l.kind {
                                LineKind::Error => p.danger,
                                LineKind::Warn => p.warn,
                                LineKind::Frontend => p.accent,
                                LineKind::Info => p.text,
                            };
                            ui.add(egui::Label::new(RichText::new(&l.text).monospace().color(color)).extend());
                        }
                    },
                );
            });
            copy
        }
    }

    /// The running program, or the last one: Stop, command line, log file.
    fn process_card(&mut self, ui: &mut egui::Ui, actions: &mut Vec<Action>) {
        let p = pal(ui);
        theme::card(ui).inner_margin(egui::Margin::symmetric(16, 12)).show(ui, |ui| {
            ui.set_width(ui.available_width());
            let Some((r, kind)) = &self.running else {
                ui.label(RichText::new("Nothing has run yet. Play a game from the library; its output and log file show here.").color(p.weak));
                return;
            };
            let live = r.exit.is_none();
            ui.horizontal(|ui| {
                theme::dot(ui, if live { p.ok } else { p.weak }, 9.0);
                ui.label(RichText::new(&r.title).size(16.0).strong().color(p.text));
                theme::pill(ui, kind.label(), p.accent);
                let secs = r.ran_for.unwrap_or_else(|| r.started.elapsed()).as_secs();
                let state = match r.exit {
                    None => format!("running, pid {}, {}", r.child.id(), theme::duration_text(secs)),
                    Some(st) => {
                        format!("{} after {}", if r.stopped { "stopped".into() } else { launcher::describe_exit(&st) }, theme::duration_text(secs))
                    }
                };
                ui.label(RichText::new(state).color(p.weak));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if live && theme::danger_button(ui, "Stop", Vec2::new(80.0, 28.0)).on_hover_text("Kill the process").clicked() {
                        actions.push(Action::Stop);
                    }
                    if let Some(f) = r.log_file.clone() {
                        if ui.add_enabled(f.is_file(), egui::Button::new("Open log file")).on_hover_text(f.display().to_string()).clicked() {
                            actions.push(Action::OpenPath(f));
                        }
                    }
                    if ui.button("Copy command line").on_hover_text("The exact command and environment of this run").clicked() {
                        self.copy_text = Some(r.command_line.clone());
                        self.status = "Command line copied".into();
                    }
                });
            });
            ui.add(egui::Label::new(RichText::new(&r.command_line).monospace().size(11.0).color(p.weak)).truncate()).on_hover_text(&r.command_line);
        });
    }

    fn log_files(&mut self, ui: &mut egui::Ui) {
        let p = pal(ui);
        ui.label(RichText::new("RUN LOGS").size(11.0).strong().color(p.weak));
        ui.add_space(2.0);
        let files = launcher::log_files(&launcher::logs_dir());
        if files.is_empty() {
            ui.label(RichText::new("No run logs yet.").color(p.weak));
            return;
        }
        let now = theme::now_secs();
        egui::ScrollArea::vertical().id_salt("log-files").auto_shrink([false; 2]).show(ui, |ui| {
            for f in files.into_iter().take(RECENT_LOGS) {
                let (name, ms) = launcher::log_file_info(&f);
                let shown = self.log_view.file.as_ref().is_some_and(|(p, _)| *p == f);
                let label = format!("{}\n{}", self.title_for_log(&name), theme::ago_text(ms / 1000, now));
                let r = ui.add_sized([ui.available_width(), 40.0], egui::SelectableLabel::new(shown, label)).on_hover_text(f.display().to_string());
                if r.clicked() {
                    match std::fs::read(&f) {
                        Ok(bytes) => {
                            let log = Log::from_text(&String::from_utf8_lossy(&bytes));
                            self.log_view.file = Some((f.clone(), log));
                            self.log_view.follow = false;
                        }
                        Err(e) => self.status = format!("{}: {e}", f.display()),
                    }
                }
                r.context_menu(|ui| {
                    if ui.button("Open in the editor").clicked() {
                        self.open_path(&f);
                        ui.close_menu();
                    }
                });
            }
        });
    }

    /// The game title of a run log name (its title id), else the name.
    fn title_for_log(&self, name: &str) -> String {
        self.games
            .iter()
            .find(|g| g.title_id.as_deref().is_some_and(|id| id.eq_ignore_ascii_case(name)))
            .map(|g| g.title.clone())
            .unwrap_or_else(|| name.replace('_', " "))
    }

    pub(crate) fn run_dialog_window(&mut self, ctx: &egui::Context) {
        let Some(RunDialog { key, options }) = self.run_dialog.as_ref() else { return };
        let Some(game) = self.game(key).cloned() else {
            self.run_dialog = None;
            return;
        };
        let mut open = true;
        let mut run = false;
        let mut cancel = false;
        let mut copy = None;
        let dbg = self.debug_dir(&game);
        let mut opts = options.clone();
        let busy = self.is_running();
        let p = theme::pal_ctx(ctx);
        let this: &App = self;
        egui::Window::new(format!("Run with options: {}", game.title))
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .default_width(600.0)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                ui.label(
                    RichText::new("Debugging options for this run, on top of the game's settings. They are kept for the next time.").color(p.weak),
                );
                ui.add_space(6.0);
                egui::Grid::new("run-with").num_columns(2).spacing([16.0, 10.0]).show(ui, |ui| {
                    ui.label("Log filter").on_hover_text("RUST_LOG of the emulator");
                    ui.horizontal(|ui| {
                        let current = run_options::LOG_PRESETS.iter().find(|(v, _)| *v == opts.rust_log.trim()).map(|(_, l)| *l).unwrap_or("Custom");
                        egui::ComboBox::from_id_salt("rw-log").selected_text(current).width(180.0).show_ui(ui, |ui| {
                            for (v, l) in run_options::LOG_PRESETS {
                                ui.selectable_value(&mut opts.rust_log, v.to_string(), l);
                            }
                        });
                        ui.add(egui::TextEdit::singleline(&mut opts.rust_log).hint_text("e.g. info,ps5gpu=debug").desired_width(200.0));
                    });
                    ui.end_row();
                    ui.label("GPU profile").on_hover_text("RPS5E_GPU_PROFILE: a [gpu] line in the log every 5 s");
                    theme::segmented(ui, &mut opts.gpu_profile, &[(0, "Off"), (1, "On"), (2, "Detailed")]);
                    ui.end_row();
                    ui.label("Overlay");
                    ui.checkbox(&mut opts.perf_overlay, "Performance overlay from the start (F10 toggles)");
                    ui.end_row();
                    ui.label("Display resolution").on_hover_text("RPS5E_DISPLAY_RESOLUTION: the TV resolution the game is told about");
                    let current = match opts.display_resolution.trim() {
                        "" => "From the game's settings",
                        v => crate::config::DISPLAY_RESOLUTIONS.iter().find(|m| m.0 == v).map(|m| m.1).unwrap_or(v),
                    };
                    egui::ComboBox::from_id_salt("rw-display-res").selected_text(current).width(180.0).show_ui(ui, |ui| {
                        ui.selectable_value(&mut opts.display_resolution, String::new(), "From the game's settings");
                        for (k, label) in crate::config::DISPLAY_RESOLUTIONS {
                            ui.selectable_value(&mut opts.display_resolution, k.to_string(), label);
                        }
                    });
                    ui.end_row();
                    ui.label("Shader compile wait").on_hover_text("RPS5E_COMPILE_WAIT_MS: how long a frame waits for shaders still compiling");
                    ui.horizontal(|ui| {
                        let mode = match opts.compile_wait.trim() {
                            "" => 0,
                            w if w.eq_ignore_ascii_case("exact") => 1,
                            _ => 2,
                        };
                        let mut m = mode;
                        theme::segmented(ui, &mut m, &[(0, "Default (3 ms)"), (1, "Exact"), (2, "Custom")]);
                        if m != mode {
                            opts.compile_wait = match m {
                                0 => String::new(),
                                1 => "exact".into(),
                                _ => "16".into(),
                            };
                        }
                        if m == 2 {
                            let mut ms: u32 = opts.compile_wait.trim().parse().unwrap_or(16);
                            if ui.add(egui::DragValue::new(&mut ms).range(0..=1000).suffix(" ms")).changed() {
                                opts.compile_wait = ms.to_string();
                            }
                        }
                    });
                    ui.end_row();
                    ui.label("HLE trace").on_hover_text("PS5_HLE_TRACE: every system call with its arguments, to a file (large, slow)");
                    ui.vertical(|ui| {
                        ui.checkbox(&mut opts.hle_trace, "Trace system library calls to a file");
                        ui.add_enabled_ui(opts.hle_trace, |ui| {
                            ui.horizontal(|ui| {
                                ui.add(
                                    egui::TextEdit::singleline(&mut opts.hle_trace_file)
                                        .hint_text(dbg.join("hle.txt").display().to_string())
                                        .desired_width(300.0),
                                );
                                ui.label("from");
                                ui.add(egui::DragValue::new(&mut opts.hle_trace_after).range(0..=3600).suffix(" s"));
                            });
                        });
                    });
                    ui.end_row();
                    ui.label("Frame dump").on_hover_text("RPS5E_DUMP_FRAMES: presented frames as image files");
                    ui.vertical(|ui| {
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut opts.dump_frames, "Save every");
                            ui.add_enabled(opts.dump_frames, egui::DragValue::new(&mut opts.dump_every).range(1..=10_000));
                            ui.label("th frame");
                        });
                        ui.add_enabled(
                            opts.dump_frames,
                            egui::TextEdit::singleline(&mut opts.dump_dir).hint_text(dbg.join("frames").display().to_string()).desired_width(300.0),
                        );
                    });
                    ui.end_row();
                });
                ui.add_space(8.0);
                let line = this.command_line(&game, Some(&opts));
                ui.label(RichText::new("COMMAND LINE").size(11.0).strong().color(p.weak));
                egui::Frame::new().fill(ui.visuals().extreme_bg_color).corner_radius(6).inner_margin(egui::Margin::same(8)).show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    match &line {
                        Ok(l) => {
                            egui::ScrollArea::vertical().max_height(90.0).show(ui, |ui| {
                                ui.add(egui::Label::new(RichText::new(l).monospace().size(11.0)).wrap());
                            });
                        }
                        Err(e) => {
                            ui.colored_label(p.danger, e);
                        }
                    }
                });
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    let ok = line.is_ok() && !busy && game.can_run();
                    let r = theme::primary_button(ui, ok, "▶  Run", Vec2::new(120.0, 34.0));
                    let r = if busy { r.on_disabled_hover_text("Another program is running") } else { r };
                    run = r.clicked();
                    if theme::button(ui, line.is_ok(), "Copy command line", Vec2::new(160.0, 34.0)).clicked() {
                        copy = line.as_ref().ok().cloned();
                    }
                    if theme::button(ui, !opts.is_default(), "Reset", Vec2::new(80.0, 34.0)).on_hover_text("No debugging options").clicked() {
                        opts = RunOptions::default();
                    }
                    cancel = theme::button(ui, true, "Cancel", Vec2::new(80.0, 34.0)).clicked();
                });
            });
        if let Some(t) = copy {
            self.copy_text = Some(t);
            self.status = "Command line copied".into();
        }
        if let Some(d) = self.run_dialog.as_mut() {
            d.options = opts.clone();
        }
        if run {
            let key = game.key();
            if opts.is_default() {
                self.settings.run_options.remove(&key);
            } else {
                self.settings.run_options.insert(key, opts.clone());
            }
            self.save_settings();
            self.run_dialog = None;
            self.launch_with(&game, Some(&opts));
            if self.is_running() {
                self.page = Page::Logs;
            }
        } else if cancel || !open {
            self.run_dialog = None;
        }
    }
}
