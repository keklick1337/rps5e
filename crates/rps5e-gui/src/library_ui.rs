// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The library page: toolbar (search, order, view, add), the games as a
//! cover grid or a list grouped by library folder, and the details panel
//! of the selected game with its actions.

use crate::app::{Action, App, Column, FilePurpose, Page};
use crate::compat::GameStatus;
use crate::library::{self, Game, PackageState};
use crate::sections::{self, Section};
use crate::settings::LibraryView;
use crate::theme::{self, pal, Palette};
use eframe::egui::{self, Color32, CornerRadius, FontId, RichText, Sense, Stroke, Vec2};

const COVER: f32 = 148.0;
const CARD_PAD: f32 = 8.0;
const DETAILS_WIDTH: f32 = 340.0;
const ROW_HEIGHT: f32 = 46.0;

/// One line of text cut to `width` with an ellipsis.
fn one_line(ui: &egui::Ui, text: &str, font: FontId, color: Color32, width: f32) -> std::sync::Arc<egui::Galley> {
    let mut job = egui::text::LayoutJob::simple_singleline(text.to_string(), font, color);
    job.wrap = egui::text::TextWrapping::truncate_at_width(width);
    ui.fonts(|f| f.layout_job(job))
}

fn status_color(s: GameStatus) -> Color32 {
    let [r, g, b] = s.color();
    Color32::from_rgb(r, g, b)
}

/// A stable colour of a game without a cover.
fn placeholder_color(title: &str) -> Color32 {
    let h = title.bytes().fold(2166136261u32, |h, b| (h ^ b as u32).wrapping_mul(16777619));
    let hue = (h % 360) as f32 / 360.0;
    egui::ecolor::HsvaGamma { h: hue, s: 0.45, v: 0.38, a: 1.0 }.into()
}

fn initials(title: &str) -> String {
    title.split_whitespace().filter_map(|w| w.chars().find(|c| c.is_alphanumeric())).take(2).collect::<String>().to_uppercase()
}

fn package_color(p: &PackageState, pal: &Palette) -> Color32 {
    match p {
        PackageState::Runnable => pal.ok,
        PackageState::NotAGame(_) => pal.warn,
        _ => pal.danger,
    }
}

impl App {
    /// The game's cover painted into `rect` (a coloured tile with its
    /// initials when it has none).
    pub(crate) fn paint_cover(&self, ui: &egui::Ui, g: &Game, rect: egui::Rect, radius: u8) {
        match g.icon.as_ref().and_then(|p| self.icons.get(p)) {
            Some(t) => {
                egui::Image::from_texture(egui::load::SizedTexture::new(t.id(), rect.size())).corner_radius(radius).paint_at(ui, rect);
            }
            None => {
                let c = placeholder_color(&g.title);
                ui.painter().rect_filled(rect, CornerRadius::same(radius), c);
                let text = if g.source.is_package() { "PKG".to_string() } else { initials(&g.title) };
                ui.painter().text(
                    rect.center(),
                    egui::Align2::CENTER_CENTER,
                    text,
                    FontId::proportional(rect.height() * 0.28),
                    Color32::from_white_alpha(220),
                );
            }
        }
    }

    pub(crate) fn library_page(&mut self, ui: &mut egui::Ui, actions: &mut Vec<Action>) {
        self.library_toolbar(ui, actions);
        ui.add_space(10.0);
        if self.games.is_empty() {
            self.empty_library(ui, actions);
            return;
        }
        let selected = self.selected_game();
        egui::SidePanel::right("details")
            .exact_width(DETAILS_WIDTH)
            .resizable(false)
            .show_separator_line(false)
            .frame(
                egui::Frame::new()
                    .fill(pal(ui).card)
                    .corner_radius(12)
                    .inner_margin(egui::Margin::same(18))
                    .stroke(Stroke::new(1.0f32, pal(ui).border)),
            )
            .show_inside(ui, |ui| match &selected {
                Some(g) => self.details(ui, g, actions),
                None => {
                    ui.add_space(80.0);
                    ui.vertical_centered(|ui| {
                        ui.label(RichText::new("No game selected").size(17.0).color(pal(ui).text));
                        ui.label(RichText::new("Click a game to see its details;\ndouble-click to play.").color(pal(ui).weak));
                    });
                }
            });
        egui::CentralPanel::default().frame(egui::Frame::new().inner_margin(egui::Margin { right: 14, ..Default::default() })).show_inside(
            ui,
            |ui| {
                let games = self.visible_games();
                if games.is_empty() {
                    ui.add_space(40.0);
                    ui.vertical_centered(|ui| {
                        ui.label(RichText::new(format!("No game matches \"{}\"", self.filter.trim())).size(16.0).color(pal(ui).weak));
                    });
                    return;
                }
                match self.settings.view {
                    LibraryView::Grid => self.grid(ui, &games, actions),
                    LibraryView::List => self.list(ui, &games, actions),
                }
            },
        );
    }

    fn library_toolbar(&mut self, ui: &mut egui::Ui, actions: &mut Vec<Action>) {
        let p = pal(ui);
        ui.horizontal(|ui| {
            let n = self.games.len();
            let sub = if self.scan_rx.is_some() { "scanning...".to_string() } else { format!("{n} game{}", if n == 1 { "" } else { "s" }) };
            theme::page_title(ui, "Library", &sub);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.menu_button(RichText::new("+  Add").strong(), |ui| {
                    ui.set_min_width(240.0);
                    let mut item = |ui: &mut egui::Ui, text: &str, hint: &str, purpose: FilePurpose| {
                        if ui.button(text).on_hover_text(hint).clicked() {
                            actions.push(Action::Pick(purpose));
                            ui.close_menu();
                        }
                    };
                    item(ui, "Library folder...", "A folder searched (with its subfolders) for games and packages", FilePurpose::AddLibraryFolder);
                    item(ui, "Game folder...", "One game folder with eboot.bin", FilePurpose::AddGameFolder);
                    item(ui, "Game file...", "eboot.bin, a package (.pkg / .ffpkg) or an executable (.elf / .self)", FilePurpose::AddGame);
                    if self.pkg_install == Some(true) {
                        ui.separator();
                        item(
                            ui,
                            "Install package...",
                            "Extract a package into the data folder's games/, which joins the library",
                            FilePurpose::InstallPackage,
                        );
                    }
                });
                if ui.button("⟳").on_hover_text("Scan the library folders again (F5)").clicked() {
                    actions.push(Action::Rescan);
                }
                if ui.button("🏆").on_hover_text("Trophies of all games").clicked() {
                    actions.push(Action::TrophyOverview);
                }
                let mut view = self.settings.view;
                if theme::segmented(ui, &mut view, &[(LibraryView::List, "List"), (LibraryView::Grid, "Grid")]) {
                    self.settings.view = view;
                    self.save_settings();
                }
                let (col, asc) = self.sort;
                egui::ComboBox::from_id_salt("sort").width(130.0).selected_text(format!("{} {}", col.label(), if asc { "⏶" } else { "⏷" })).show_ui(
                    ui,
                    |ui| {
                        for c in Column::ALL {
                            if ui.selectable_label(c == col, c.label()).clicked() {
                                self.sort = if c == col { (c, !asc) } else { (c, true) };
                            }
                        }
                    },
                );
                ui.add(
                    egui::TextEdit::singleline(&mut self.filter)
                        .hint_text(RichText::new("🔍 Search by name or title ID").color(p.weak))
                        .desired_width(240.0)
                        .margin(egui::Margin::symmetric(8, 5)),
                );
            });
        });
    }

    fn empty_library(&mut self, ui: &mut egui::Ui, actions: &mut Vec<Action>) {
        ui.add_space(50.0);
        ui.vertical_centered(|ui| {
            if self.scan_rx.is_some() {
                ui.spinner();
                ui.label("Scanning the library folders...");
                return;
            }
            if let Some(t) = theme::logo(ui.ctx()) {
                ui.add(egui::Image::from_texture(egui::load::SizedTexture::new(t.id(), Vec2::splat(96.0))));
            }
            ui.add_space(8.0);
            ui.label(RichText::new("Your library is empty").size(22.0).strong());
            let roots = self.settings.library_roots();
            let text = if roots.is_empty() {
                "Add a folder with games: it is searched, with its subfolders, for game folders (eboot.bin and sce_sys/param.json) and game packages (.pkg, .ffpkg).".to_string()
            } else {
                let list: Vec<String> = roots.iter().map(|r| r.display().to_string()).collect();
                format!("No games found in {}. Put game folders (eboot.bin and sce_sys/) or packages there, or add them below.", list.join(", "))
            };
            ui.add(egui::Label::new(RichText::new(text).color(pal(ui).weak)).wrap());
            ui.add_space(14.0);
            if theme::primary_button(ui, true, "Add a folder with games...", Vec2::new(260.0, 38.0)).clicked() {
                actions.push(Action::Pick(FilePurpose::AddLibraryFolder));
            }
            ui.add_space(4.0);
            if theme::button(ui, true, "Add one game...", Vec2::new(260.0, 32.0)).clicked() {
                actions.push(Action::Pick(FilePurpose::AddGame));
            }
            if self.settings.folder_rows().iter().any(|r| !r.enabled) && theme::button(ui, true, "Library folders...", Vec2::new(260.0, 32.0)).on_hover_text("Some library folders are turned off").clicked() {
                self.tab = crate::app::Tab::Library;
                actions.push(Action::Show(Page::Settings));
            }
        });
    }

    /// A section's header: the folder's name, its path and game count, and
    /// the section menu.
    fn section_header(&self, ui: &mut egui::Ui, section: &Section, shown: usize, total: usize, actions: &mut Vec<Action>) {
        let p = pal(ui);
        ui.horizontal(|ui| {
            let (name, path) = match section {
                Section::Folder(d) => {
                    (d.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| d.display().to_string()), d.display().to_string())
                }
                Section::Added => ("Added games".to_string(), "added one by one".to_string()),
            };
            let r = ui.add(egui::Label::new(RichText::new(name).size(15.5).strong().color(p.text)).sense(Sense::click()));
            r.on_hover_text(section.detail()).context_menu(|ui| self.section_menu(ui, section, actions));
            let count = if shown == total { format!("{total}") } else { format!("{shown} of {total}") };
            theme::pill(ui, &count, p.accent);
            ui.add(egui::Label::new(RichText::new(path).small().color(p.weak)).truncate());
            ui.menu_button(RichText::new("•••").size(10.0).color(p.weak), |ui| self.section_menu(ui, section, actions))
                .response
                .on_hover_text("Folder menu");
        });
    }

    fn section_menu(&self, ui: &mut egui::Ui, section: &Section, actions: &mut Vec<Action>) {
        if let Section::Folder(d) = section {
            menu_item(ui, d.is_dir(), "Open folder", Action::OpenPath(d.clone()), actions);
        }
        menu_item(ui, true, "Rescan", Action::Rescan, actions);
        ui.separator();
        if let Section::Folder(d) = section {
            menu_item(ui, true, "Turn off (keep in the list)", Action::SetFolderEnabled(d.clone(), false), actions);
        }
        let text = match section {
            Section::Folder(_) => "Remove folder from library...",
            Section::Added => "Remove added games from library...",
        };
        menu_item(ui, !self.is_running(), text, Action::RemoveSection(section.clone()), actions);
    }

    fn grid(&mut self, ui: &mut egui::Ui, games: &[Game], actions: &mut Vec<Action>) {
        let groups = sections::group(&self.settings, games);
        let all = sections::group(&self.settings, &self.games);
        egui::ScrollArea::vertical().auto_shrink([false; 2]).show(ui, |ui| {
            for (section, games) in &groups {
                let total = all.iter().find(|(s, _)| s == section).map(|(_, v)| v.len()).unwrap_or(games.len());
                self.section_header(ui, section, games.len(), total, actions);
                ui.add_space(4.0);
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing = Vec2::new(14.0, 14.0);
                    for g in games {
                        self.grid_card(ui, g, actions);
                    }
                });
                ui.add_space(18.0);
            }
        });
    }

    fn grid_card(&self, ui: &mut egui::Ui, g: &Game, actions: &mut Vec<Action>) {
        let p = pal(ui);
        let key = g.key();
        let selected = self.selected.as_ref() == Some(&key);
        let running = self.running_game() == Some(key.as_str());
        let size = Vec2::new(COVER + 2.0 * CARD_PAD, COVER + 2.0 * CARD_PAD + 44.0);
        let (rect, resp) = ui.allocate_exact_size(size, Sense::click());
        let bg = if selected {
            p.accent.gamma_multiply(0.18)
        } else if resp.hovered() {
            p.hover
        } else {
            p.card
        };
        ui.painter().rect_filled(rect, CornerRadius::same(10), bg);
        let stroke = if selected { Stroke::new(2.0f32, p.accent) } else { Stroke::new(1.0f32, p.border) };
        ui.painter().rect_stroke(rect, CornerRadius::same(10), stroke, egui::StrokeKind::Inside);
        let cover = egui::Rect::from_min_size(rect.min + Vec2::splat(CARD_PAD), Vec2::splat(COVER));
        self.paint_cover(ui, g, cover, 7);
        if running {
            let badge = egui::Rect::from_min_size(cover.left_top() + Vec2::new(6.0, 6.0), Vec2::new(78.0, 20.0));
            ui.painter().rect_filled(badge, CornerRadius::same(10), p.ok);
            ui.painter().text(badge.center(), egui::Align2::CENTER_CENTER, "▶ Running", FontId::proportional(11.5), Color32::WHITE);
        } else if let Some(state) = &g.package {
            let badge = egui::Rect::from_min_size(cover.left_top() + Vec2::new(6.0, 6.0), Vec2::new(40.0, 20.0));
            ui.painter().rect_filled(badge, CornerRadius::same(10), package_color(state, &p));
            ui.painter().text(badge.center(), egui::Align2::CENTER_CENTER, "PKG", FontId::proportional(11.5), Color32::WHITE);
        }
        let text_w = COVER;
        let title = one_line(ui, &g.title, FontId::proportional(14.0), p.text, text_w);
        let y = cover.bottom() + 7.0;
        ui.painter().galley(egui::pos2(cover.left(), y), title, p.text);
        let status = self.status_of(g);
        let dot = egui::pos2(cover.left() + 4.0, y + 27.0);
        ui.painter().circle_filled(dot, 4.0, status_color(status));
        let meta = format!("{}  {}", g.title_id.as_deref().unwrap_or("-"), g.version.as_deref().map(short_version).unwrap_or_default());
        let meta = one_line(ui, &meta, FontId::proportional(11.5), p.weak, text_w - 14.0);
        ui.painter().galley(egui::pos2(cover.left() + 12.0, y + 20.0), meta, p.weak);
        let resp = resp.on_hover_text(format!("{}\n{}\nStatus: {}", g.title, g.source.path().display(), status.label()));
        self.item_response(resp, g, actions);
    }

    fn item_response(&self, resp: egui::Response, g: &Game, actions: &mut Vec<Action>) {
        let key = g.key();
        if resp.clicked() {
            actions.push(Action::Select(key.clone()));
        }
        if resp.double_clicked() && g.can_run() && !self.is_running() {
            actions.push(Action::Launch(key.clone()));
        }
        resp.context_menu(|ui| self.context_menu(ui, g, actions));
    }

    fn context_menu(&self, ui: &mut egui::Ui, g: &Game, actions: &mut Vec<Action>) {
        let key = g.key();
        let running = self.running_game() == Some(key.as_str());
        let busy = self.is_running();
        actions.push(Action::Select(key.clone()));
        ui.set_min_width(210.0);
        menu_item(ui, !busy && g.can_run(), "▶  Play", Action::Launch(key.clone()), actions);
        menu_item(ui, !busy && g.can_run(), "Run with options...", Action::RunWith(key.clone()), actions);
        if running {
            menu_item(ui, true, "■  Stop", Action::Stop, actions);
        }
        if g.source.is_package() && self.pkg_install == Some(true) {
            menu_item(ui, !busy && installable(g), "Install package", Action::Install(key.clone()), actions);
        }
        ui.separator();
        menu_item(ui, !running && g.can_run(), "Game settings...", Action::EditConfig(key.clone()), actions);
        if g.title_id.is_some() {
            menu_item(ui, true, "Input profile...", Action::InputProfile(key.clone()), actions);
        }
        if g.title_id.as_deref().is_some_and(crate::cheats::is_supported_title_id) {
            menu_item(ui, true, "Cheats...", Action::Cheats(key.clone()), actions);
        }
        if !g.source.is_package() && crate::trophies::has_trophy_data(&g.dir) {
            menu_item(ui, true, "Trophies...", Action::Trophies(key.clone()), actions);
        }
        if g.title_id.is_some() {
            ui.menu_button("Shaders", |ui| {
                menu_item(ui, !busy, "Compile now", Action::Precompile(key.clone()), actions);
                menu_item(ui, !busy, "Recompile (clear cache)", Action::Recompile(key.clone()), actions);
                menu_item(ui, !busy, "Clear cache", Action::ClearShaderCache(key.clone()), actions);
                ui.separator();
                menu_item(ui, !busy, "Export cache...", Action::ExportShaderCache(key.clone()), actions);
                menu_item(ui, !busy, "Import cache...", Action::ImportShaderCache(key.clone()), actions);
            });
            if let Some(id) = &g.title_id {
                ui.menu_button("Compatibility", |ui| {
                    let current = self.status_of(g);
                    for s in GameStatus::ALL {
                        let text = if current == s { format!("✔  {}", s.label()) } else { format!("     {}", s.label()) };
                        menu_item(ui, true, &text, Action::SetStatus(id.clone(), s), actions);
                    }
                });
            }
        }
        ui.separator();
        menu_item(ui, g.dir.is_dir(), "Open game folder", Action::OpenFolder(key.clone()), actions);
        menu_item(ui, g.can_run(), "Copy command line", Action::CopyCommand(key.clone()), actions);
        menu_item(ui, !running && !self.save_data_dirs(g).is_empty(), "Remove save data...", Action::RemoveSaveData(key.clone()), actions);
        menu_item(ui, true, "Report issue...", Action::ReportIssue(key.clone()), actions);
        ui.separator();
        menu_item(ui, !running, "Remove from library...", Action::RemoveGame(key.clone()), actions);
    }

    fn list(&mut self, ui: &mut egui::Ui, games: &[Game], actions: &mut Vec<Action>) {
        let p = pal(ui);
        // Columns after the name: (column, width).
        let cols = [(Column::Serial, 96.0), (Column::Version, 96.0), (Column::Size, 76.0), (Column::Status, 104.0), (Column::LastRun, 100.0)];
        let fixed: f32 = cols.iter().map(|c| c.1).sum::<f32>() + 44.0;
        let all = sections::group(&self.settings, &self.games);
        egui::ScrollArea::vertical().auto_shrink([false; 2]).show(ui, |ui| {
            let width = ui.available_width();
            let name_w = (width - fixed - 16.0).max(160.0);
            // Header: click to sort.
            let (hrect, _) = ui.allocate_exact_size(Vec2::new(width, 24.0), Sense::hover());
            let mut x = hrect.left() + 44.0;
            let mut header = |ui: &mut egui::Ui, col: Column, w: f32, x: f32| {
                let r = egui::Rect::from_min_size(egui::pos2(x, hrect.top()), Vec2::new(w, 24.0));
                let resp = ui.interact(r, ui.id().with(("sort", col.label())), Sense::click());
                let arrow = if self.sort.0 == col {
                    if self.sort.1 {
                        " ⏶"
                    } else {
                        " ⏷"
                    }
                } else {
                    ""
                };
                let color = if self.sort.0 == col { p.text } else { p.weak };
                ui.painter().text(r.left_center(), egui::Align2::LEFT_CENTER, format!("{}{arrow}", col.label()), FontId::proportional(12.5), color);
                if resp.clicked() {
                    self.sort = if self.sort.0 == col { (col, !self.sort.1) } else { (col, true) };
                }
            };
            header(ui, Column::Name, name_w, x);
            x += name_w;
            for (c, w) in cols {
                header(ui, c, w, x);
                x += w;
            }
            ui.painter().hline(hrect.x_range(), hrect.bottom(), Stroke::new(1.0f32, p.border));
            for (section, games) in sections::group(&self.settings, games) {
                ui.add_space(8.0);
                let total = all.iter().find(|(s, _)| *s == section).map(|(_, v)| v.len()).unwrap_or(games.len());
                self.section_header(ui, &section, games.len(), total, actions);
                for g in &games {
                    self.list_row(ui, g, width, name_w, &cols, actions);
                }
            }
        });
    }

    fn list_row(&self, ui: &mut egui::Ui, g: &Game, width: f32, name_w: f32, cols: &[(Column, f32)], actions: &mut Vec<Action>) {
        let p = pal(ui);
        let key = g.key();
        let selected = self.selected.as_ref() == Some(&key);
        let (rect, resp) = ui.allocate_exact_size(Vec2::new(width, ROW_HEIGHT), Sense::click());
        if selected {
            ui.painter().rect_filled(rect, CornerRadius::same(8), p.accent.gamma_multiply(0.2));
        } else if resp.hovered() {
            ui.painter().rect_filled(rect, CornerRadius::same(8), p.hover);
        }
        let icon = egui::Rect::from_min_size(rect.min + Vec2::new(4.0, 5.0), Vec2::splat(ROW_HEIGHT - 10.0));
        self.paint_cover(ui, g, icon, 5);
        let mid = rect.center().y;
        let mut x = rect.left() + 44.0;
        let running = self.running_game() == Some(key.as_str());
        let name = if running { format!("▶ {}", g.title) } else { g.title.clone() };
        let galley = one_line(ui, &name, FontId::proportional(14.0), p.text, name_w - 12.0);
        let name_color = if running { p.ok } else { p.text };
        ui.painter().galley(egui::pos2(x, mid - galley.size().y / 2.0), galley, name_color);
        x += name_w;
        for (c, w) in cols {
            let (text, color) = match c {
                Column::Serial => (g.title_id.clone().unwrap_or_else(|| "-".into()), p.weak),
                Column::Version => match &g.package {
                    Some(st) => (st.label().split(',').next().unwrap_or("package").to_string(), package_color(st, &p)),
                    None => (g.version.clone().unwrap_or_default(), p.weak),
                },
                Column::Size => (self.sizes.get(&key).map(|s| library::format_size(*s)).unwrap_or_else(|| "-".into()), p.weak),
                Column::Status => {
                    let s = self.status_of(g);
                    ui.painter().circle_filled(egui::pos2(x + 4.0, mid), 4.0, status_color(s));
                    (format!("     {}", s.label()), p.text)
                }
                Column::LastRun => match self.settings.last_runs.get(&key) {
                    Some(r) => (theme::ago_text(r.ended, theme::now_secs()), if r.ok { p.weak } else { p.danger }),
                    None => ("never".into(), p.weak),
                },
                Column::Name => (String::new(), p.weak),
            };
            let galley = one_line(ui, &text, FontId::proportional(12.5), color, w - 8.0);
            ui.painter().galley(egui::pos2(x, mid - galley.size().y / 2.0), galley, color);
            x += w;
        }
        let resp = resp.on_hover_text(g.source.path().display().to_string());
        self.item_response(resp, g, actions);
    }

    /// The details panel of the selected game.
    fn details(&mut self, ui: &mut egui::Ui, g: &Game, actions: &mut Vec<Action>) {
        let p = pal(ui);
        let key = g.key();
        let busy = self.is_running();
        let running_this = self.running_game() == Some(key.as_str());
        let banner = self.banner(g);
        egui::ScrollArea::vertical().auto_shrink([false; 2]).show(ui, |ui| {
            let w = ui.available_width();
            // The game's background picture with its cover on it; without
            // one, the cover alone.
            let (hero, _) = ui.allocate_exact_size(Vec2::new(w, (w * 9.0 / 16.0).round()), Sense::hover());
            let cover_size = 84.0;
            match &banner {
                Some(t) => {
                    egui::Image::from_texture(egui::load::SizedTexture::new(t.id(), hero.size())).corner_radius(10).paint_at(ui, hero);
                    let cover = egui::Rect::from_min_size(hero.left_bottom() + Vec2::new(12.0, -cover_size - 12.0), Vec2::splat(cover_size));
                    ui.painter().rect_filled(cover.expand(2.0), CornerRadius::same(9), Color32::from_black_alpha(160));
                    self.paint_cover(ui, g, cover, 7);
                }
                None => {
                    ui.painter().rect_filled(hero, CornerRadius::same(10), ui.visuals().extreme_bg_color);
                    let side = hero.height() - 20.0;
                    self.paint_cover(ui, g, egui::Rect::from_center_size(hero.center(), Vec2::splat(side)), 8);
                }
            }
            ui.add_space(10.0);
            ui.add(egui::Label::new(RichText::new(&g.title).size(20.0).strong().color(p.text)).wrap());
            ui.horizontal_wrapped(|ui| {
                if let Some(id) = &g.title_id {
                    theme::pill(ui, id, p.accent);
                }
                if let Some(v) = &g.version {
                    theme::pill(ui, &format!("v{v}"), p.weak);
                }
                let s = self.status_of(g);
                theme::pill(ui, s.label(), status_color(s));
                if self.configs.has(&key) {
                    theme::pill(ui, "own settings", p.warn);
                }
            });
            ui.add_space(10.0);
            // Play / Stop and Run with.
            ui.horizontal(|ui| {
                let play_w = w - 120.0;
                if running_this {
                    if theme::danger_button(ui, "■  Stop", Vec2::new(play_w, 40.0)).clicked() {
                        actions.push(Action::Stop);
                    }
                } else {
                    let r = theme::primary_button(ui, !busy && g.can_run(), "▶  Play", Vec2::new(play_w, 40.0));
                    let r = if busy {
                        r.on_disabled_hover_text("Another program is running (see the sidebar)")
                    } else if let Some(st) = &g.package {
                        r.on_disabled_hover_text(st.detail())
                    } else {
                        r
                    };
                    if r.clicked() {
                        actions.push(Action::Launch(key.clone()));
                    }
                }
                let rw = theme::button(ui, !busy && g.can_run(), "Run with...", Vec2::new(112.0, 40.0))
                    .on_hover_text("Run with debugging options: log level, GPU profile, HLE trace, frame dump, compile wait, overlay");
                if rw.clicked() {
                    actions.push(Action::RunWith(key.clone()));
                }
            });
            if let Some(o) = self.settings.run_options.get(&key).filter(|o| !o.is_default()) {
                ui.label(RichText::new(format!("Run with: {}", o.summary())).small().color(p.weak));
            }
            if g.source.is_package() && self.pkg_install == Some(true) {
                let r = theme::button(ui, !busy && installable(g), "Install package", Vec2::new(w, 30.0))
                    .on_hover_text("Extract it into the data folder's games/ (that folder joins the library)");
                if r.clicked() {
                    actions.push(Action::Install(key.clone()));
                }
            }
            ui.add_space(12.0);
            self.info_grid(ui, g, actions);
            ui.add_space(8.0);
            group_title(ui, "Game");
            let mut buttons: Vec<(&str, bool, Action, &str)> = vec![(
                "Settings...",
                g.can_run() && !running_this,
                Action::EditConfig(key.clone()),
                "This game's own settings (also F1 in the game)",
            )];
            if g.title_id.is_some() {
                buttons.push(("Input profile...", true, Action::InputProfile(key.clone()), "Key bindings and controller settings of this game"));
            }
            if g.title_id.as_deref().is_some_and(crate::cheats::is_supported_title_id) {
                buttons.push(("Cheats...", true, Action::Cheats(key.clone()), "Cheat file of this game (applied at start)"));
            }
            if !g.source.is_package() && crate::trophies::has_trophy_data(&g.dir) {
                buttons.push(("Trophies...", true, Action::Trophies(key.clone()), "The game's trophies and the ones earned"));
            }
            button_grid(ui, buttons, actions);
            if g.title_id.is_some() {
                group_title(ui, "Shaders");
                button_grid(
                    ui,
                    vec![
                        ("Compile now", !busy, Action::Precompile(key.clone()), "Build the game's shaders now instead of while playing"),
                        ("Recompile", !busy, Action::Recompile(key.clone()), "Clear the game's shader and pipeline caches and build them again"),
                        ("Clear cache", !busy, Action::ClearShaderCache(key.clone()), "Delete the game's compiled shaders; they are built again before the next start"),
                        ("Export...", !busy, Action::ExportShaderCache(key.clone()), "Save the shader cache as a portable file"),
                        ("Import...", !busy, Action::ImportShaderCache(key.clone()), "Merge a shader cache file into the game's cache"),
                    ],
                    actions,
                );
            }
            group_title(ui, "Files and help");
            button_grid(
                ui,
                vec![
                    ("Open folder", g.dir.is_dir(), Action::OpenFolder(key.clone()), "Open the game's folder"),
                    (
                        "Copy command",
                        g.can_run(),
                        Action::CopyCommand(key.clone()),
                        "Copy the command line that runs the game (with its Run with options)",
                    ),
                    (
                        "Remove saves...",
                        !running_this && !self.save_data_dirs(g).is_empty(),
                        Action::RemoveSaveData(key.clone()),
                        "Delete the game's save data",
                    ),
                    ("Report issue...", true, Action::ReportIssue(key.clone()), "Open a prefilled issue report in the browser"),
                ],
                actions,
            );
            ui.add_space(8.0);
            self.compat_editor(ui, g, actions);
            ui.add_space(6.0);
            let r = ui.add_enabled(!running_this, egui::Button::new(RichText::new("Remove from library...").color(p.danger)).frame(false));
            if r.on_hover_text("Hide the game in the library; its files stay on disk").clicked() {
                actions.push(Action::RemoveGame(key.clone()));
            }
        });
    }

    fn info_grid(&self, ui: &mut egui::Ui, g: &Game, actions: &mut Vec<Action>) {
        let p = pal(ui);
        let key = g.key();
        egui::Grid::new("details-info").num_columns(2).spacing([12.0, 6.0]).show(ui, |ui| {
            let row = |ui: &mut egui::Ui, k: &str, v: String| {
                ui.label(RichText::new(k).color(p.weak));
                ui.add(egui::Label::new(RichText::new(v).color(p.text)).truncate());
                ui.end_row();
            };
            if let Some(v) = g.app_version.as_ref().filter(|v| Some(*v) != g.version.as_ref()) {
                row(ui, "App version", v.clone());
            }
            if let Some(f) = &g.firmware {
                row(ui, "Firmware", f.clone());
            }
            row(ui, "Size", self.sizes.get(&key).map(|s| library::format_size(*s)).unwrap_or_else(|| "measuring...".into()));
            let kind = match &g.package {
                Some(st) => st.label().to_string(),
                None if matches!(g.source, library::GameSource::Executable(_)) => "executable".into(),
                None => "game folder".into(),
            };
            row(ui, "Type", kind);
            ui.label(RichText::new("Location").color(p.weak));
            let path = g.source.path().display().to_string();
            let r = ui.add(egui::Label::new(RichText::new(&path).color(p.accent)).truncate().sense(Sense::click()));
            if r.on_hover_text(format!("{path}\nClick to open the folder")).clicked() {
                actions.push(Action::OpenFolder(key.clone()));
            }
            ui.end_row();
            ui.label(RichText::new("Settings").color(p.weak));
            ui.horizontal(|ui| {
                if self.configs.has(&key) {
                    ui.label("own");
                    if ui.small_button("use global").on_hover_text("Clear this game's settings").clicked() {
                        actions.push(Action::ClearConfig(key.clone()));
                    }
                } else {
                    ui.label("global");
                }
            });
            ui.end_row();
            ui.label(RichText::new("Last run").color(p.weak));
            match self.settings.last_runs.get(&key) {
                Some(r) => {
                    let text = format!("{}, {} ({})", theme::ago_text(r.ended, theme::now_secs()), theme::duration_text(r.seconds), r.result);
                    ui.add(egui::Label::new(RichText::new(text).color(if r.ok { p.text } else { p.danger })).truncate());
                }
                None => {
                    ui.label(RichText::new("never").color(p.weak));
                }
            }
            ui.end_row();
        });
    }

    fn compat_editor(&mut self, ui: &mut egui::Ui, g: &Game, actions: &mut Vec<Action>) {
        let Some(id) = g.title_id.clone() else { return };
        let p = pal(ui);
        group_title(ui, "Compatibility (your notes)");
        let current = self.status_of(g);
        let mut s = current;
        egui::ComboBox::from_id_salt(("status", &id))
            .width(ui.available_width())
            .selected_text(RichText::new(s.label()).color(status_color(s)))
            .show_ui(ui, |ui| {
                for v in GameStatus::ALL {
                    ui.selectable_value(&mut s, v, RichText::new(v.label()).color(status_color(v)));
                }
            });
        if s != current {
            actions.push(Action::SetStatus(id.clone(), s));
        }
        let stored = self.compat.find(&id).map(|e| e.comment.clone()).unwrap_or_default();
        let text = self.comment_edit.entry(g.key()).or_insert_with(|| stored.clone());
        let r = ui.add(
            egui::TextEdit::multiline(text)
                .hint_text(RichText::new("Notes: what works, what breaks").color(p.weak))
                .desired_rows(2)
                .desired_width(f32::INFINITY),
        );
        if r.lost_focus() && *text != stored {
            actions.push(Action::SetComment(id, text.clone()));
        } else if !r.has_focus() && *text != stored {
            *text = stored;
        }
    }
}

/// `01.018.000` -> `v01.018`: the patch number only when it is not zero.
fn short_version(v: &str) -> String {
    let parts: Vec<&str> = v.split('.').collect();
    match parts.as_slice() {
        [a, b, c] if c.trim_start_matches('0').is_empty() => format!("v{a}.{b}"),
        _ => format!("v{v}"),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn short_versions() {
        assert_eq!(super::short_version("01.018.000"), "v01.018");
        assert_eq!(super::short_version("01.004.003"), "v01.004.003");
        assert_eq!(super::short_version("01.000"), "v01.000");
    }
}

fn installable(g: &Game) -> bool {
    g.package.as_ref().is_some_and(|p| matches!(p, PackageState::Runnable | PackageState::NotAGame(_)))
}

fn group_title(ui: &mut egui::Ui, title: &str) {
    ui.add_space(4.0);
    ui.label(RichText::new(title.to_uppercase()).size(11.0).strong().color(pal(ui).weak));
}

/// Buttons two per row filling the width.
fn button_grid(ui: &mut egui::Ui, buttons: Vec<(&str, bool, Action, &str)>, actions: &mut Vec<Action>) {
    let w = (ui.available_width() - ui.spacing().item_spacing.x) / 2.0;
    for pair in buttons.chunks(2) {
        ui.horizontal(|ui| {
            for (text, enabled, action, hint) in pair {
                if theme::button(ui, *enabled, text, Vec2::new(w, 30.0)).on_hover_text(*hint).clicked() {
                    actions.push(action.clone());
                }
            }
        });
    }
}

pub(crate) fn menu_item(ui: &mut egui::Ui, enabled: bool, text: &str, a: Action, actions: &mut Vec<Action>) {
    if ui.add_enabled(enabled, egui::Button::new(text)).clicked() {
        actions.push(a);
        ui.close_menu();
    }
}
