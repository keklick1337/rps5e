// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Per-game windows: game configuration, cheats (local file, optional
//! online collection), trophy viewer and the trophy overview.

use crate::app::{App, FilePurpose};
use crate::cheats::{self, RemoteFile};
use crate::config::{EmulatorSettings, GameConfig};
use crate::library::Game;
use crate::trophies::{self, TrophySet};
use eframe::egui::{self, Color32, RichText, Sense, TextureHandle, Vec2};
use ps5hle::np_trophy2::package::Progress;
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver};

// ---------------------------------------------------------------------------
// Game configuration
// ---------------------------------------------------------------------------

pub(crate) struct GameConfigDialog {
    key: String,
    title_id: String,
    name: String,
    settings: EmulatorSettings,
    had_config: bool,
}

impl GameConfigDialog {
    /// Start from the game's configuration, else a copy of the global one.
    pub fn new(app: &App, g: &Game) -> Self {
        Self {
            key: g.key(),
            title_id: g.title_id.clone().unwrap_or_default(),
            name: g.title.clone(),
            settings: app.effective(g),
            had_config: app.configs.has(&g.key()),
        }
    }
}

// ---------------------------------------------------------------------------
// Cheats
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum CheatTab {
    Local,
    Online,
}

pub(crate) struct CheatsDialog {
    name: String,
    title_id: String,
    version: String,
    process: String,
    path: Option<PathBuf>,
    tab: CheatTab,
    local_doc: Option<Value>,
    local_data: Vec<u8>,
    local_readable: bool,
    selection: Vec<(String, bool)>,
    status: String,
    online: bool,
    catalog_rx: Option<Receiver<(Vec<RemoteFile>, String)>>,
    files: Vec<RemoteFile>,
    remote_status: String,
    picked: Option<usize>,
    preview_rx: Option<Receiver<Result<Vec<u8>, String>>>,
    preview_doc: Option<Value>,
    preview_status: String,
    importable: bool,
    confirm_replace: bool,
}

impl CheatsDialog {
    pub fn new(app: &App, g: &Game) -> Self {
        let cfg = app.effective(g);
        let title_id = g.title_id.clone().unwrap_or_default().trim().to_ascii_uppercase();
        let process = Path::new(&cfg.elf).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "eboot.bin".into());
        let mut d = Self {
            name: g.title.clone(),
            path: cheats::plan_path(&cfg.effective_data_dir(), &title_id),
            title_id,
            version: g.app_version.clone().unwrap_or_default(),
            process,
            tab: CheatTab::Local,
            local_doc: None,
            local_data: Vec::new(),
            local_readable: false,
            selection: Vec::new(),
            status: String::new(),
            online: app.settings.network.online_cheats,
            catalog_rx: None,
            files: Vec::new(),
            remote_status: String::new(),
            picked: None,
            preview_rx: None,
            preview_doc: None,
            preview_status: String::new(),
            importable: false,
            confirm_replace: false,
        };
        d.load();
        d
    }

    fn load(&mut self) {
        self.local_doc = None;
        self.local_data.clear();
        self.local_readable = false;
        self.selection.clear();
        let Some(path) = self.path.clone() else {
            self.status = "Unsupported title ID.".into();
            return;
        };
        match std::fs::read(&path) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                self.local_readable = true;
                self.status = format!("No local cheat file: {}", path.display());
            }
            Err(e) => self.status = format!("Could not read cheat file: {e}"),
            Ok(data) => {
                self.local_readable = true;
                match cheats::parse_document(&data) {
                    Ok(doc) => {
                        self.selection = cheats::mods(&doc);
                        self.status = format!("Loaded {} cheat(s) from {}.", self.selection.len(), path.display());
                        self.local_doc = Some(doc);
                    }
                    Err(e) => self.status = format!("{e}\n{}", path.display()),
                }
                self.local_data = data;
            }
        }
    }

    fn save_selection(&mut self) {
        let (Some(doc), Some(path)) = (&self.local_doc, &self.path) else { return };
        let enabled: Vec<bool> = self.selection.iter().map(|s| s.1).collect();
        let Some(next) = cheats::with_selection(doc, &enabled) else { return };
        match cheats::save_document(path, &next, &self.local_data) {
            Ok(data) => {
                self.local_data = data;
                self.local_doc = Some(next);
                self.status = "Cheat selection saved. Changes take effect on the next game launch.".into();
            }
            Err(e) => self.status = e,
        }
    }

    fn load_remote(&mut self) {
        let id = self.title_id.clone();
        let (tx, rx) = channel();
        self.catalog_rx = Some(rx);
        self.files.clear();
        self.picked = None;
        self.preview_doc = None;
        self.preview_status.clear();
        self.importable = false;
        self.remote_status = "Loading online cheats...".into();
        let _ = std::thread::Builder::new().name("rps5e-gui-cheats".into()).spawn(move || {
            let _ = tx.send(cheats::load_catalog(&id));
        });
    }

    fn preview(&mut self, index: usize) {
        self.picked = Some(index);
        self.preview_doc = None;
        self.importable = false;
        let Some(file) = self.files.get(index).cloned().filter(RemoteFile::is_supported) else { return };
        self.preview_status = "Loading cheat preview...".into();
        let (tx, rx) = channel();
        self.preview_rx = Some(rx);
        let _ = std::thread::Builder::new().name("rps5e-gui-cheat".into()).spawn(move || {
            let _ = tx.send(cheats::fetch_file(&file));
        });
    }

    fn poll(&mut self) {
        if let Some(rx) = &self.catalog_rx {
            if let Ok((mut files, err)) = rx.try_recv() {
                self.catalog_rx = None;
                cheats::sort_files(&mut files, &self.version);
                self.remote_status = if files.is_empty() && err.is_empty() {
                    format!("No online cheats found for {}.", self.title_id)
                } else {
                    format!("{} online file(s) for {}.", files.len(), self.title_id)
                };
                if !err.is_empty() {
                    self.remote_status.push_str(&format!("\nCould not load the complete collection: {err}\nUse Refresh to retry."));
                }
                self.files = files;
            }
        }
        if let Some(rx) = &self.preview_rx {
            if let Ok(res) = rx.try_recv() {
                self.preview_rx = None;
                self.show_candidate(res.and_then(|d| cheats::parse_document(&d)));
            }
        }
    }

    /// A downloaded or picked file, checked before it can be imported.
    fn show_candidate(&mut self, doc: Result<Value, String>) {
        match doc {
            Err(e) => {
                self.preview_doc = None;
                self.importable = false;
                self.preview_status = format!("Could not preview this file: {e}");
            }
            Ok(doc) => {
                let check = cheats::validate_import(&doc, &self.title_id, &self.version, &self.process);
                let name = doc.get("name").and_then(Value::as_str).unwrap_or("").to_string();
                let ver = doc.get("version").and_then(Value::as_str).unwrap_or("").to_string();
                self.preview_status = format!(
                    "{name}\nCredits: {} - Version: {ver}\n\n{}",
                    cheats::credits(&doc),
                    match &check {
                        Ok(()) => "Import, then select the cheats to enable. Changes take effect on the next game launch.".to_string(),
                        Err(e) => e.clone(),
                    }
                );
                self.importable = check.is_ok();
                self.preview_doc = Some(doc);
            }
        }
    }

    /// A local JSON file picked by the user.
    pub fn import_file(&mut self, path: &Path) {
        self.tab = CheatTab::Online;
        self.picked = None;
        self.show_candidate(std::fs::read(path).map_err(|e| e.to_string()).and_then(|d| cheats::parse_document(&d)));
    }

    fn import(&mut self) {
        let (Some(doc), Some(path)) = (self.preview_doc.clone(), self.path.clone()) else { return };
        if let Err(e) = cheats::validate_import(&doc, &self.title_id, &self.version, &self.process) {
            self.preview_status = e;
            return;
        }
        if !self.local_readable {
            self.preview_status = "Could not read the local cheat file. Reload it before importing.".into();
            return;
        }
        let n = cheats::mods(&doc).len();
        // Imported cheats start unchecked.
        let Some(sel) = cheats::with_selection(&doc, &vec![false; n]) else { return };
        match cheats::save_document(&path, &sel, &self.local_data) {
            Ok(_) => {
                self.load();
                self.status = format!("Imported {n} cheat(s). Select cheats and apply your selection for the next game launch.");
                self.tab = CheatTab::Local;
            }
            Err(e) => self.preview_status = e,
        }
    }
}

// ---------------------------------------------------------------------------
// Trophies
// ---------------------------------------------------------------------------

pub(crate) struct TrophyDialog {
    title: String,
    icon: Option<PathBuf>,
    sets: Vec<TrophySet>,
    errors: Vec<String>,
    tab: usize,
    textures: HashMap<(u32, i32, bool), Option<TextureHandle>>,
    inspect: Option<(usize, i32, bool)>,
}

impl TrophyDialog {
    pub fn load(app: &App, g: &Game) -> Result<Self, String> {
        let cfg = app.effective(g);
        let id = g.title_id.clone().unwrap_or_default();
        let (sets, errors) = trophies::load_game(&g.dir, &id, &cfg.effective_data_dir(), cfg.console_language);
        if sets.is_empty() {
            return Err(if errors.is_empty() { "No trophy package found in sce_sys/trophy2.".into() } else { errors.join("\n") });
        }
        Ok(Self { title: g.title.clone(), icon: g.icon.clone(), sets, errors, tab: 0, textures: HashMap::new(), inspect: None })
    }

    fn texture(&mut self, ctx: &egui::Context, set: usize, id: i32, unlocked: bool) -> Option<TextureHandle> {
        let label = self.sets[set].label;
        let key = (label, id, unlocked);
        if let Some(t) = self.textures.get(&key) {
            return t.clone();
        }
        let tex = self.sets[set].package.trophies.get(&id).and_then(|t| trophies::decode_png(&t.icon_png)).map(|(w, h, mut px)| {
            if !unlocked {
                trophies::grayscale(&mut px);
            }
            let img = egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &px);
            ctx.load_texture(format!("trophy-{label}-{id}-{unlocked}"), img, egui::TextureOptions::LINEAR)
        });
        self.textures.insert(key, tex.clone());
        tex
    }
}

pub(crate) struct TrophyOverview {
    rows: Vec<(String, String, String, Option<PathBuf>, Progress)>,
    totals: Progress,
}

impl TrophyOverview {
    pub fn load(app: &App) -> Self {
        let mut rows = Vec::new();
        let mut totals = Progress::default();
        for g in app.games.iter().filter(|g| !g.source.is_package()) {
            let cfg = app.effective(g);
            let id = g.title_id.clone().unwrap_or_default();
            let (sets, _) = trophies::load_game(&g.dir, &id, &cfg.effective_data_dir(), cfg.console_language);
            let p = trophies::game_progress(&sets);
            if p.total == 0 {
                continue;
            }
            totals.earned += p.earned;
            for grade in 1..5 {
                totals.earned_grade[grade] += p.earned_grade[grade];
            }
            rows.push((g.key(), g.title.clone(), id, g.icon.clone(), p));
        }
        Self { rows, totals }
    }
}

fn cup(ui: &mut egui::Ui, grade: i32, size: f32) {
    let [r, g, b] = trophies::grade_color(grade);
    let color = Color32::from_rgb(r, g, b);
    let (rect, resp) = ui.allocate_exact_size(Vec2::splat(size), Sense::hover());
    let p = ui.painter();
    let u = size / 20.0;
    let at = |x: f32, y: f32| rect.min + Vec2::new(x * u, y * u);
    p.add(egui::Shape::convex_polygon(vec![at(5.0, 2.0), at(15.0, 2.0), at(14.0, 9.0), at(10.0, 12.0), at(6.0, 9.0)], color, egui::Stroke::NONE));
    p.line_segment([at(10.0, 12.0), at(10.0, 16.0)], egui::Stroke::new(1.6 * u, color));
    p.rect_filled(egui::Rect::from_min_max(at(6.0, 16.0), at(14.0, 18.5)), u, color);
    resp.on_hover_text(trophies::grade_name(grade));
}

fn grade_counts(ui: &mut egui::Ui, p: &Progress, all: bool) {
    for grade in 1..=4 {
        if !all && p.total_grade[grade] == 0 {
            continue;
        }
        cup(ui, grade as i32, 18.0);
        ui.label(p.earned_grade[grade].to_string());
    }
}

// ---------------------------------------------------------------------------
// Windows
// ---------------------------------------------------------------------------

impl App {
    pub(crate) fn game_config_window(&mut self, ctx: &egui::Context) {
        let Some(d) = self.game_config.as_mut() else { return };
        let mut open = true;
        let mut action = None;
        let mut clear = false;
        egui::Window::new(format!("Game config - {}", d.name)).open(&mut open).default_size([700.0, 600.0]).vscroll(true).show(ctx, |ui| {
            ui.label(RichText::new(format!("{}  {}", d.title_id, d.key)).small());
            if !d.had_config {
                ui.label(RichText::new("Starts from the global settings; saving gives the game its own configuration.").italics());
            }
            crate::settings_ui::emulator_form(ui, &mut d.settings, "game", true);
            ui.separator();
            ui.horizontal(|ui| {
                if ui.add_enabled(d.settings.validate().is_ok(), egui::Button::new("Save")).clicked() {
                    action = Some(true);
                }
                if ui.button("Cancel").clicked() {
                    action = Some(false);
                }
                if d.had_config && ui.button("Clear game config").on_hover_text("Use the global settings").clicked() {
                    clear = true;
                }
            });
        });
        if clear {
            self.confirm = Some(crate::app::Confirm {
                title: "Clear game config".into(),
                text: "Clear this game's config and use global settings?".into(),
                action: crate::app::ConfirmAction::ClearGameConfig(d.key.clone()),
            });
            self.game_config = None;
            return;
        }
        match action {
            Some(true) => {
                let d = self.game_config.take().expect("dialog");
                self.configs.games.insert(d.key, GameConfig { title_id: d.title_id, name: d.name, settings: d.settings });
                self.save_configs();
                self.status = "Game config saved".into();
            }
            Some(false) => self.game_config = None,
            None if !open => self.game_config = None,
            None => {}
        }
    }

    pub(crate) fn cheats_window(&mut self, ctx: &egui::Context) {
        let Some(d) = self.cheats.as_mut() else { return };
        d.poll();
        if d.catalog_rx.is_some() || d.preview_rx.is_some() {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
        let mut open = true;
        let mut pick_file = false;
        egui::Window::new(format!("Cheats (experimental) - {}", d.name)).open(&mut open).default_size([720.0, 560.0]).show(ctx, |ui| {
            let ver = if d.version.is_empty() { "Unknown" } else { d.version.as_str() };
            ui.label(format!("{} - Installed version: {ver} - {}", d.title_id, d.process));
            ui.horizontal(|ui| {
                ui.selectable_value(&mut d.tab, CheatTab::Local, "Local");
                if ui.selectable_value(&mut d.tab, CheatTab::Online, "Import").clicked()
                    && d.online
                    && d.files.is_empty()
                    && d.catalog_rx.is_none()
                {
                    d.load_remote();
                }
            });
            ui.separator();
            match d.tab {
                CheatTab::Local => {
                    egui::ScrollArea::vertical().max_height(360.0).show(ui, |ui| {
                        for (name, on) in &mut d.selection {
                            ui.checkbox(on, name.as_str());
                        }
                    });
                    ui.label(&d.status);
                    ui.horizontal(|ui| {
                        if ui.button("Reload").clicked() {
                            d.load();
                        }
                        if ui.add_enabled(!d.selection.is_empty(), egui::Button::new("Apply selection")).clicked() {
                            d.save_selection();
                        }
                    });
                }
                CheatTab::Online => {
                    ui.horizontal(|ui| {
                        if ui.button("Import a file...").clicked() {
                            pick_file = true;
                        }
                        if d.online {
                            if ui.add_enabled(d.catalog_rx.is_none(), egui::Button::new("Refresh")).clicked() {
                                d.load_remote();
                            }
                        } else {
                            ui.label(RichText::new("Online collection: enable it in Settings > Interface > Network.").small());
                        }
                    });
                    if d.online {
                        ui.hyperlink_to("Cheats by HEN Cheats Collection and its contributors", cheats::REPOSITORY_URL);
                        ui.label(RichText::new("Only JSON cheats are supported. MC4 and SHN files are shown disabled.").small());
                        let mut chosen = None;
                        egui::ScrollArea::vertical().id_salt("remote").max_height(180.0).show(ui, |ui| {
                            egui::Grid::new("remote-files").striped(true).num_columns(3).show(ui, |ui| {
                                ui.strong("Version");
                                ui.strong("Format");
                                ui.strong("File");
                                ui.end_row();
                                for (i, f) in d.files.iter().enumerate() {
                                    let v = if f.version == d.version { format!("{} (installed)", f.version) } else { f.version.clone() };
                                    ui.label(v);
                                    ui.label(f.format.to_ascii_uppercase());
                                    let r = ui
                                        .add_enabled(f.is_supported(), egui::SelectableLabel::new(d.picked == Some(i), &f.name))
                                        .on_hover_text(&f.title);
                                    if r.clicked() {
                                        chosen = Some(i);
                                    }
                                    ui.end_row();
                                }
                            });
                        });
                        if let Some(i) = chosen {
                            d.preview(i);
                        }
                        ui.label(&d.remote_status);
                    }
                    if let Some(doc) = &d.preview_doc {
                        ui.separator();
                        egui::ScrollArea::vertical().id_salt("preview").max_height(120.0).show(ui, |ui| {
                            for (name, _) in cheats::mods(doc) {
                                ui.label(name);
                            }
                        });
                    }
                    ui.label(&d.preview_status);
                    if d.confirm_replace {
                        ui.colored_label(Color32::YELLOW, format!("Replace the local cheat file and selections for {}? Imported cheats start unchecked.", d.title_id));
                        ui.horizontal(|ui| {
                            if ui.button("Replace").clicked() {
                                d.confirm_replace = false;
                                d.import();
                            }
                            if ui.button("Cancel").clicked() {
                                d.confirm_replace = false;
                            }
                        });
                    } else if ui.add_enabled(d.importable, egui::Button::new("Import to local")).clicked() {
                        if d.path.as_ref().is_some_and(|p| p.exists()) {
                            d.confirm_replace = true;
                        } else {
                            d.import();
                        }
                    }
                }
            }
        });
        if !open {
            self.cheats = None;
        } else if pick_file {
            self.open_file_dialog(FilePurpose::ImportCheat);
        }
    }

    pub(crate) fn trophy_window(&mut self, ctx: &egui::Context) {
        let Some(d) = self.trophies.as_mut() else { return };
        let mut open = true;
        let game_icon = d.icon.as_ref().and_then(|p| self.icons.get(p)).cloned();
        egui::Window::new(format!("Trophy viewer - {}", d.title)).open(&mut open).default_size([820.0, 600.0]).show(ctx, |ui| {
            if d.sets.len() > 1 {
                ui.horizontal(|ui| {
                    for (i, s) in d.sets.iter().enumerate() {
                        ui.selectable_value(&mut d.tab, i, &s.package.title);
                    }
                });
            }
            for e in &d.errors {
                ui.colored_label(Color32::from_rgb(255, 110, 110), e);
            }
            let set = d.tab.min(d.sets.len() - 1);
            let p = d.sets[set].progress();
            ui.horizontal(|ui| {
                if let Some(t) = &game_icon {
                    ui.add(egui::Image::from_texture(egui::load::SizedTexture::new(t.id(), Vec2::splat(56.0))));
                }
                ui.heading(&d.sets[set].package.title);
                ui.separator();
                ui.label(format!("Progress {}%", p.percentage()));
                ui.label(format!("Earned {}/{}", p.earned, p.total));
                grade_counts(ui, &p, false);
            });
            ui.add(egui::ProgressBar::new(p.percentage() as f32 / 100.0));
            ui.label(format!("All trophies: {}", p.total));
            let ids: Vec<i32> = d.sets[set].package.trophies.keys().copied().collect();
            let mut clicked = None;
            egui::ScrollArea::vertical().auto_shrink([false; 2]).show(ui, |ui| {
                for id in ids {
                    let (unlocked, hidden_locked, name, grade, detail, date) = {
                        let s = &d.sets[set];
                        let t = &s.package.trophies[&id];
                        let unlocked = s.unlocks.unlocked.contains(&id);
                        let masked = t.hidden && !unlocked;
                        let date = trophies::earned_date(id, &s.unlocks).or_else(|| unlocked.then(|| "Earned".to_string()));
                        (unlocked, masked, trophies::trophy_name(id, t, masked), t.grade, t.description.clone(), date)
                    };
                    let tex = if hidden_locked { None } else { d.texture(ctx, set, id, unlocked) };
                    let resp = ui
                        .horizontal(|ui| {
                            match tex {
                                Some(t) => {
                                    ui.add(egui::Image::from_texture(egui::load::SizedTexture::new(t.id(), Vec2::splat(64.0))));
                                }
                                None => {
                                    let (r, _) = ui.allocate_exact_size(Vec2::splat(64.0), Sense::hover());
                                    ui.painter().rect_filled(r, 8.0, Color32::from_gray(70));
                                    ui.painter().text(r.center(), egui::Align2::CENTER_CENTER, "?", egui::FontId::proportional(28.0), Color32::LIGHT_GRAY);
                                }
                            }
                            ui.vertical(|ui| {
                                ui.label(RichText::new(name).strong().size(16.0));
                                if !hidden_locked {
                                    ui.horizontal(|ui| {
                                        cup(ui, grade, 16.0);
                                        ui.label(trophies::grade_name(grade));
                                    });
                                    ui.label(detail);
                                }
                            });
                            if let Some(date) = date {
                                ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| ui.label(date));
                            }
                        })
                        .response
                        .interact(Sense::click());
                    if resp.clicked() {
                        clicked = Some(id);
                    }
                    ui.separator();
                }
            });
            if let Some(id) = clicked {
                d.inspect = Some((set, id, false));
            }
        });
        // Trophy details.
        if let Some((set, id, revealed)) = d.inspect {
            let s = &d.sets[set];
            let t = &s.package.trophies[&id];
            let unlocked = s.unlocks.unlocked.contains(&id);
            let masked = t.hidden && !unlocked && !revealed;
            let mut keep = true;
            let mut reveal = false;
            let name = trophies::trophy_name(id, t, masked);
            let date = trophies::earned_date(id, &s.unlocks);
            let (grade, description, reward, has_reward, hidden) = (t.grade, t.description.clone(), t.reward.clone(), t.has_reward, t.hidden);
            let tex = if masked { None } else { d.texture(ctx, set, id, unlocked) };
            egui::Window::new("Trophy details").open(&mut keep).collapsible(false).show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if let Some(t) = tex {
                        ui.add(egui::Image::from_texture(egui::load::SizedTexture::new(t.id(), Vec2::splat(128.0))));
                    }
                    ui.vertical(|ui| {
                        ui.heading(&name);
                        if hidden && !masked {
                            ui.label("Hidden trophy");
                        }
                        if masked && ui.button("Show hidden trophy").clicked() {
                            reveal = true;
                        }
                    });
                });
                egui::Grid::new("trophy-detail").num_columns(2).show(ui, |ui| {
                    ui.label("Grade");
                    ui.label(if masked { "Hidden trophy" } else { trophies::grade_name(grade) });
                    ui.end_row();
                    ui.label("Status");
                    ui.label(if unlocked { "Earned" } else { "Not earned" });
                    ui.end_row();
                    ui.label("Earned date");
                    ui.label(if unlocked { date.unwrap_or_else(|| "-".into()) } else { "-".into() });
                    ui.end_row();
                    ui.label("Details");
                    ui.label(if masked || description.is_empty() { "-".to_string() } else { description });
                    ui.end_row();
                    if has_reward && !reward.is_empty() {
                        ui.label("Reward");
                        ui.label(if masked { "-".to_string() } else { reward });
                        ui.end_row();
                    }
                });
            });
            d.inspect = if !keep { None } else if reveal { Some((set, id, true)) } else { d.inspect };
        }
        if !open {
            self.trophies = None;
        }
    }

    pub(crate) fn trophy_overview_window(&mut self, ctx: &egui::Context) {
        let Some(o) = &self.trophy_overview else { return };
        let mut open = true;
        let mut chosen = None;
        egui::Window::new("Trophies").open(&mut open).default_size([700.0, 560.0]).show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Trophies");
                ui.separator();
                ui.label(format!("Total {}", o.totals.earned));
                grade_counts(ui, &o.totals, true);
            });
            ui.separator();
            if o.rows.is_empty() {
                ui.label("No games with trophies found in the game folders.");
                return;
            }
            egui::ScrollArea::vertical().auto_shrink([false; 2]).show(ui, |ui| {
                for (key, name, id, icon, p) in &o.rows {
                    let resp = ui
                        .horizontal(|ui| {
                            match icon.as_ref().and_then(|i| self.icons.get(i)) {
                                Some(t) => {
                                    ui.add(egui::Image::from_texture(egui::load::SizedTexture::new(t.id(), Vec2::splat(64.0))));
                                }
                                None => {
                                    let (r, _) = ui.allocate_exact_size(Vec2::splat(64.0), Sense::hover());
                                    ui.painter().rect_filled(r, 6.0, Color32::from_gray(60));
                                }
                            }
                            ui.vertical(|ui| {
                                ui.label(RichText::new(name).strong().size(16.0));
                                ui.label(id);
                            });
                            ui.vertical(|ui| {
                                ui.label(RichText::new(format!("{}%", p.percentage())).size(20.0));
                                ui.label(format!("Earned {}/{}", p.earned, p.total));
                                ui.add(egui::ProgressBar::new(p.percentage() as f32 / 100.0).desired_width(130.0));
                            });
                            ui.horizontal(|ui| grade_counts(ui, p, p.total_grade[1] != 0));
                        })
                        .response
                        .interact(Sense::click());
                    if resp.clicked() {
                        chosen = Some(key.clone());
                    }
                    ui.separator();
                }
            });
        });
        if !open {
            self.trophy_overview = None;
        }
        if let Some(k) = chosen {
            if let Some(g) = self.game(&k).cloned() {
                match TrophyDialog::load(self, &g) {
                    Ok(d) => self.trophies = Some(d),
                    Err(e) => self.status = e,
                }
            }
        }
    }
}
