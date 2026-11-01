// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! On-screen keyboard of the system text-entry dialog (libSceImeDialog),
//! drawn with egui over the game in the game window.
//!
//! [`install`] registers the dialog's visibility callback: it only stores
//! the running session's id in `ps5gpu::menu` (an atomic, safe on guest
//! threads), which hands the window's input to the in-game UI and makes it
//! call [`ImeDialogUi::show`] every frame while the menu is closed. Without
//! the callback (no window, unattended runs) the HLE side types a default
//! text and accepts by itself.
//!
//! Each frame reads the session ([`ps5hle::ime_dialog::host_snapshot`]),
//! applies the input through the `host_*` edit functions (the HLE side
//! filters characters by the dialog type and limits the length) and draws
//! from a fresh snapshot: title, the text field (placeholder, bullets for a
//! password, the caret), a `length / max` counter, the key grid and the
//! button hints.
//!
//! Input:
//! - physical keyboard: typed text is inserted, Backspace deletes, Left /
//!   Right move the caret, Enter accepts (inserts a line break in a
//!   multi-line dialog; Ctrl+Enter accepts there), Escape cancels;
//! - controller ([`ps5input::menu::Nav`] steps): D-pad / left stick move on
//!   the grid, Cross presses the key, Circle cancels, Triangle types a
//!   space, Square deletes, L1 / R1 move the caret, L2 is Shift, R2 and
//!   Options accept;
//! - mouse: clicking a key presses it.

use eframe::egui::{self, Color32, FontId, Pos2, Rect, Sense, Stroke, Vec2};
use ps5hle::ime_common::{
    HostSnapshot, DISABLE_DEVICE_CONTROLLER, DISABLE_DEVICE_EXT_KEYBOARD, OPTION_MULTILINE, OPTION_NO_AUTO_CAPITALIZE,
    OPTION_PASSWORD, OPTION_USE_OVER_2K, TYPE_DEFAULT, TYPE_NUMBER,
};
use ps5input::menu::Nav;

/// `RPS5E_IME_HEADLESS=1`: the dialog finishes by itself even with a
/// window (the default text is typed and accepted); `0`: the dialog waits
/// for the user even in a scripted or replayed run.
pub const HEADLESS_ENV: &str = "RPS5E_IME_HEADLESS";

fn flag(v: Option<String>) -> bool {
    v.is_some_and(|v| !v.is_empty() && v != "0")
}

fn set(v: Option<String>) -> bool {
    v.is_some_and(|v| !v.is_empty())
}

/// Whether a dialog waits for the user in a run with these variables:
/// not when forced headless, without host input, or while a pad script,
/// an input recording or its replay drives the game (those keep their
/// timing only when the dialog finishes by itself).
pub fn waits_for_user(var: impl Fn(&str) -> Option<String>) -> bool {
    if var(HEADLESS_ENV).as_deref() == Some("0") {
        return true;
    }
    !(flag(var(HEADLESS_ENV))
        || flag(var("RPS5E_NO_HOST_INPUT"))
        || set(var("RPS5E_PAD_SCRIPT"))
        || set(var("RPS5E_INPUT_RECORD"))
        || set(var("RPS5E_INPUT_REPLAY")))
}

fn on_visibility(visible: bool, generation: u64) {
    ps5gpu::menu::set_system_dialog(visible, generation);
}

/// Attach the on-screen keyboard to the dialog (with the in-game menu,
/// which draws it). Returns whether the dialog now waits for the user.
pub fn install() -> bool {
    if !waits_for_user(|k| std::env::var(k).ok()) {
        log::info!("text input dialogs finish by themselves in this run");
        return false;
    }
    ps5hle::ime_dialog::set_visibility_callback(Some(on_visibility));
    true
}

// ---------------------------------------------------------------------------
// Key grid
// ---------------------------------------------------------------------------

/// What a key does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyAction {
    /// Type the character (letters follow Shift).
    Char(char),
    Shift,
    /// Switch between letters and symbols.
    Symbols,
    Space,
    Backspace,
    CaretLeft,
    CaretRight,
    Accept,
    Cancel,
}

/// A key and the grid columns it covers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Key {
    pub action: KeyAction,
    pub span: u8,
}

const fn k(action: KeyAction, span: u8) -> Key {
    Key { action, span }
}

/// The key sets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Page {
    Letters,
    Symbols,
    Number,
}

/// Rows of keys; every row covers `columns` grid columns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub columns: u8,
    pub rows: Vec<Vec<Key>>,
}

fn char_row(chars: &str) -> Vec<Key> {
    chars.chars().map(|c| k(KeyAction::Char(c), 1)).collect()
}

impl Layout {
    pub fn for_page(page: Page) -> Layout {
        use KeyAction::*;
        match page {
            Page::Number => Layout {
                columns: 3,
                rows: vec![
                    char_row("123"),
                    char_row("456"),
                    char_row("789"),
                    char_row("-0."),
                    vec![k(Char(','), 1), k(Backspace, 2)],
                    vec![k(Cancel, 1), k(Accept, 2)],
                ],
            },
            Page::Letters | Page::Symbols => {
                let chars: [&str; 4] = if page == Page::Letters {
                    ["1234567890", "qwertyuiop", "asdfghjkl'", "zxcvbnm,.-"]
                } else {
                    ["1234567890", "!@#$%^&*()", "-_=+[]{}\\|", ";:'\",./?<>"]
                };
                let mut rows: Vec<Vec<Key>> = chars.iter().map(|r| char_row(r)).collect();
                rows.push(vec![k(Shift, 2), k(Symbols, 2), k(Space, 4), k(Backspace, 2)]);
                rows.push(vec![k(CaretLeft, 1), k(CaretRight, 1), k(Cancel, 4), k(Accept, 4)]);
                Layout { columns: 10, rows }
            }
        }
    }

    /// First grid column of key `index` in `row`.
    fn start(&self, row: usize, index: usize) -> u8 {
        self.rows[row][..index].iter().map(|k| k.span).sum()
    }

    /// The key of `row` covering grid column `column` (the last one when
    /// the row is shorter).
    fn at_column(&self, row: usize, column: u8) -> usize {
        let mut start = 0u8;
        for (i, key) in self.rows[row].iter().enumerate() {
            if column < start + key.span {
                return i;
            }
            start += key.span;
        }
        self.rows[row].len() - 1
    }

    pub fn key(&self, sel: Selection) -> Key {
        self.rows[sel.row][sel.index]
    }

    /// Where `action` is (the first such key).
    pub fn find(&self, action: KeyAction) -> Option<Selection> {
        self.rows.iter().enumerate().find_map(|(row, keys)| {
            keys.iter().position(|k| k.action == action).map(|index| Selection { row, index, column: self.start(row, index) })
        })
    }
}

/// The highlighted key: row, index in the row, and the grid column that
/// up/down moves keep (so passing a wide key comes back to the same
/// column).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Selection {
    pub row: usize,
    pub index: usize,
    pub column: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dir {
    Up,
    Down,
    Left,
    Right,
}

impl Selection {
    /// Clamp to `layout` (after a page change).
    pub fn fit(self, layout: &Layout) -> Selection {
        let row = self.row.min(layout.rows.len() - 1);
        let column = self.column.min(layout.columns - 1);
        Selection { row, index: layout.at_column(row, column), column }
    }

    /// One step in `dir`, wrapping around the grid.
    pub fn step(self, layout: &Layout, dir: Dir) -> Selection {
        let s = self.fit(layout);
        let rows = layout.rows.len();
        match dir {
            Dir::Up | Dir::Down => {
                let row = if dir == Dir::Up { (s.row + rows - 1) % rows } else { (s.row + 1) % rows };
                Selection { row, index: layout.at_column(row, s.column), column: s.column }
            }
            Dir::Left | Dir::Right => {
                let n = layout.rows[s.row].len();
                let index = if dir == Dir::Left { (s.index + n - 1) % n } else { (s.index + 1) % n };
                Selection { row: s.row, index, column: layout.start(s.row, index) }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Input mapping
// ---------------------------------------------------------------------------

/// What one input asks of the dialog.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Command {
    /// Move the highlight on the grid.
    Move(Dir),
    /// Press the highlighted key.
    Press,
    /// Type text as it is (the physical keyboard).
    Insert(String),
    Space,
    Backspace,
    Caret(i32),
    Shift,
    Accept,
    Cancel,
}

/// The command of a controller step.
pub fn nav_command(n: Nav) -> Option<Command> {
    Some(match n {
        Nav::Up => Command::Move(Dir::Up),
        Nav::Down => Command::Move(Dir::Down),
        Nav::Left => Command::Move(Dir::Left),
        Nav::Right => Command::Move(Dir::Right),
        Nav::Accept => Command::Press,
        Nav::Back => Command::Cancel,
        Nav::Triangle => Command::Space,
        Nav::Square => Command::Backspace,
        Nav::PrevTab => Command::Caret(-1),
        Nav::NextTab => Command::Caret(1),
        Nav::L2 => Command::Shift,
        Nav::R2 | Nav::Close => Command::Accept,
    })
}

/// The commands of a frame's keyboard events (egui-winit sends a key
/// event, then the typed text).
pub fn keyboard_commands(events: &[egui::Event], multiline: bool) -> Vec<Command> {
    let mut out = Vec::new();
    for e in events {
        match e {
            egui::Event::Text(t) | egui::Event::Paste(t) if !t.is_empty() => out.push(Command::Insert(t.clone())),
            egui::Event::Key { key, pressed: true, modifiers, .. } => match key {
                egui::Key::Backspace => out.push(Command::Backspace),
                egui::Key::ArrowLeft => out.push(Command::Caret(-1)),
                egui::Key::ArrowRight => out.push(Command::Caret(1)),
                egui::Key::Escape => out.push(Command::Cancel),
                egui::Key::Enter if multiline && !(modifiers.ctrl || modifiers.command) => out.push(Command::Insert("\n".into())),
                egui::Key::Enter => out.push(Command::Accept),
                _ => {}
            },
            _ => {}
        }
    }
    out
}

/// The session the keyboard edits (the HLE dialog; a fake one in tests).
pub trait Session {
    fn insert(&mut self, generation: u64, text: &[u16]) -> bool;
    fn backspace(&mut self, generation: u64) -> bool;
    fn move_cursor(&mut self, generation: u64, delta: i32) -> bool;
    fn accept(&mut self, generation: u64) -> bool;
    fn cancel(&mut self, generation: u64) -> bool;
}

/// The running libSceImeDialog session.
pub struct HleSession;

impl Session for HleSession {
    fn insert(&mut self, generation: u64, text: &[u16]) -> bool {
        ps5hle::ime_dialog::host_insert_text(generation, text)
    }
    fn backspace(&mut self, generation: u64) -> bool {
        ps5hle::ime_dialog::host_backspace(generation)
    }
    fn move_cursor(&mut self, generation: u64, delta: i32) -> bool {
        ps5hle::ime_dialog::host_move_cursor(generation, delta)
    }
    fn accept(&mut self, generation: u64) -> bool {
        ps5hle::ime_dialog::host_accept(generation)
    }
    fn cancel(&mut self, generation: u64) -> bool {
        ps5hle::ime_dialog::host_cancel(generation)
    }
}

/// Shift: off, for the next letter, or locked (pressed twice).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ShiftState {
    #[default]
    Off,
    Once,
    Locked,
}

impl ShiftState {
    fn next(self) -> Self {
        match self {
            ShiftState::Off => ShiftState::Once,
            ShiftState::Once => ShiftState::Locked,
            ShiftState::Locked => ShiftState::Off,
        }
    }
}

// ---------------------------------------------------------------------------
// The keyboard
// ---------------------------------------------------------------------------

/// The on-screen keyboard's state between frames.
#[derive(Debug)]
pub struct ImeDialogUi {
    /// Session the state belongs to (0: none yet).
    generation: u64,
    pub page: Page,
    pub shift: ShiftState,
    pub sel: Selection,
    layout: Layout,
    nav: Vec<Nav>,
    /// The last insert was refused (full or not allowed): the counter
    /// flashes.
    refused: bool,
}

impl Default for ImeDialogUi {
    fn default() -> Self {
        ImeDialogUi {
            generation: 0,
            page: Page::Letters,
            shift: ShiftState::Off,
            sel: Selection::default(),
            layout: Layout::for_page(Page::Letters),
            nav: Vec::new(),
            refused: false,
        }
    }
}

impl ImeDialogUi {
    /// Queue controller steps for the next frame.
    pub fn queue_nav(&mut self, steps: &[Nav]) {
        self.nav.extend_from_slice(steps);
    }

    /// Start over for a new session.
    fn begin(&mut self, snap: &HostSnapshot) {
        self.generation = snap.generation;
        self.page = if snap.ty == TYPE_NUMBER { Page::Number } else { Page::Letters };
        self.layout = Layout::for_page(self.page);
        // A name or sentence starts with a capital letter unless the game
        // says otherwise.
        let autocap = snap.ty == TYPE_DEFAULT && snap.option & (OPTION_NO_AUTO_CAPITALIZE | OPTION_PASSWORD) == 0 && snap.text.is_empty();
        self.shift = if autocap { ShiftState::Once } else { ShiftState::Off };
        self.sel = Selection { row: 1.min(self.layout.rows.len() - 1), index: 0, column: 0 };
        self.nav.clear();
        self.refused = false;
    }

    fn set_page(&mut self, page: Page) {
        self.page = page;
        self.layout = Layout::for_page(page);
        self.sel = self.sel.fit(&self.layout);
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// The character a key types now.
    fn typed(&self, c: char) -> char {
        if self.shift != ShiftState::Off { c.to_ascii_uppercase() } else { c }
    }

    fn insert(&mut self, session: &mut dyn Session, generation: u64, text: &str) {
        let units: Vec<u16> = text.encode_utf16().collect();
        self.refused = !session.insert(generation, &units);
    }

    /// Press a key.
    pub fn press(&mut self, action: KeyAction, session: &mut dyn Session, generation: u64) {
        match action {
            KeyAction::Char(c) => {
                let c = self.typed(c);
                self.insert(session, generation, c.encode_utf8(&mut [0; 4]));
                if self.shift == ShiftState::Once && c.is_alphabetic() {
                    self.shift = ShiftState::Off;
                }
            }
            KeyAction::Shift => self.shift = self.shift.next(),
            KeyAction::Symbols => self.set_page(if self.page == Page::Symbols { Page::Letters } else { Page::Symbols }),
            KeyAction::Space => self.insert(session, generation, " "),
            KeyAction::Backspace => {
                session.backspace(generation);
                self.refused = false;
            }
            KeyAction::CaretLeft => {
                session.move_cursor(generation, -1);
            }
            KeyAction::CaretRight => {
                session.move_cursor(generation, 1);
            }
            KeyAction::Accept => {
                session.accept(generation);
            }
            KeyAction::Cancel => {
                session.cancel(generation);
            }
        }
    }

    /// Carry out one command.
    pub fn apply(&mut self, cmd: Command, session: &mut dyn Session, generation: u64) {
        match cmd {
            Command::Move(d) => self.sel = self.sel.step(&self.layout, d),
            Command::Press => {
                let key = self.layout.key(self.sel.fit(&self.layout));
                self.press(key.action, session, generation);
            }
            Command::Insert(t) => self.insert(session, generation, &t),
            Command::Space => self.press(KeyAction::Space, session, generation),
            Command::Backspace => self.press(KeyAction::Backspace, session, generation),
            Command::Caret(d) => {
                session.move_cursor(generation, d);
            }
            Command::Shift => self.press(KeyAction::Shift, session, generation),
            Command::Accept => self.press(KeyAction::Accept, session, generation),
            Command::Cancel => self.press(KeyAction::Cancel, session, generation),
        }
    }

    /// One frame of the running HLE dialog (nothing when none runs).
    pub fn show(&mut self, ctx: &egui::Context) {
        let Some(snap) = ps5hle::ime_dialog::host_snapshot() else {
            self.nav.clear();
            return;
        };
        let mut session = HleSession;
        if self.frame_input(ctx, &snap, &mut session) {
            match ps5hle::ime_dialog::host_snapshot() {
                Some(fresh) => self.draw(ctx, &fresh, &mut session),
                None => self.nav.clear(),
            }
        } else {
            self.draw(ctx, &snap, &mut session);
        }
    }

    /// Apply this frame's keyboard events and controller steps; whether
    /// anything was applied.
    pub fn frame_input(&mut self, ctx: &egui::Context, snap: &HostSnapshot, session: &mut dyn Session) -> bool {
        if snap.generation != self.generation {
            self.begin(snap);
        }
        let mut cmds = Vec::new();
        if snap.disable_device & DISABLE_DEVICE_EXT_KEYBOARD == 0 {
            let events = ctx.input(|i| i.events.clone());
            cmds = keyboard_commands(&events, snap.option & OPTION_MULTILINE != 0);
        }
        let steps = std::mem::take(&mut self.nav);
        if snap.disable_device & DISABLE_DEVICE_CONTROLLER == 0 {
            cmds.extend(steps.into_iter().filter_map(nav_command));
        }
        let any = !cmds.is_empty();
        for c in cmds {
            self.apply(c, session, snap.generation);
        }
        any
    }

    /// Draw the dialog for `snap`; clicked keys act on `session`.
    pub fn draw(&mut self, ctx: &egui::Context, snap: &HostSnapshot, session: &mut dyn Session) {
        if snap.generation != self.generation {
            self.begin(snap);
        }
        let screen = ctx.screen_rect();
        egui::Area::new(egui::Id::new("ime-backdrop")).order(egui::Order::Background).fixed_pos(screen.min).show(ctx, |ui| {
            ui.painter().rect_filled(screen, 0.0, Color32::from_black_alpha(140));
            ui.allocate_rect(screen, Sense::click());
        });
        let geo = Geometry::new(screen, snap, &self.layout);
        let mut pressed = None;
        egui::Area::new(egui::Id::new("ime-panel")).order(egui::Order::Foreground).fixed_pos(geo.panel.min).show(ctx, |ui| {
            pressed = self.paint(ui, &geo, snap);
            ui.allocate_rect(geo.panel, Sense::hover());
        });
        if let Some((sel, action)) = pressed {
            self.sel = sel;
            self.press(action, session, snap.generation);
        }
    }

    /// Paint the panel; returns a clicked key.
    fn paint(&self, ui: &mut egui::Ui, geo: &Geometry, snap: &HostSnapshot) -> Option<(Selection, KeyAction)> {
        let p = ui.painter().clone();
        p.rect_filled(geo.panel, 14.0, PANEL);
        p.rect_stroke(geo.panel, 14.0, Stroke::new(1.0f32, PANEL_EDGE), egui::StrokeKind::Inside);
        let s = geo.scale;
        let inner = geo.panel.shrink(geo.pad);
        // Title.
        let title = String::from_utf16_lossy(&snap.title);
        let title = if title.trim().is_empty() { "Enter text".to_string() } else { title };
        p.text(Pos2::new(inner.min.x, inner.min.y), egui::Align2::LEFT_TOP, title, FontId::proportional(19.0 * s), TEXT);
        // Text field.
        let field = geo.field;
        p.rect_filled(field, 8.0, FIELD);
        p.rect_stroke(field, 8.0, Stroke::new(1.5f32, ACCENT), egui::StrokeKind::Inside);
        let text = String::from_utf16_lossy(&snap.text);
        let caret_units = (snap.cursor as usize).min(snap.text.len());
        let caret_chars = String::from_utf16_lossy(&snap.text[..caret_units]).chars().count();
        let shown = if snap.option & OPTION_PASSWORD != 0 { "\u{2022}".repeat(text.chars().count()) } else { text.clone() };
        let font = FontId::proportional(20.0 * s);
        let text_rect = field.shrink2(Vec2::new(12.0 * s, 8.0 * s));
        let multiline = snap.option & OPTION_MULTILINE != 0;
        let wrap = if multiline { text_rect.width() } else { f32::INFINITY };
        let galley = p.layout(shown.clone(), font.clone(), TEXT, wrap);
        // A single line keeps the caret in view: the text slides left.
        let caret = galley.pos_from_ccursor(egui::text::CCursor::new(caret_chars));
        let shift_x = if multiline { 0.0 } else { (caret.min.x - text_rect.width() + 4.0).max(0.0) };
        let origin = Pos2::new(text_rect.min.x - shift_x, if multiline { text_rect.min.y } else { text_rect.center().y - galley.size().y / 2.0 });
        let clip = p.with_clip_rect(text_rect.expand(2.0));
        if shown.is_empty() {
            let placeholder = String::from_utf16_lossy(&snap.placeholder);
            if !placeholder.is_empty() {
                clip.text(origin + Vec2::new(4.0 * s, 0.0), egui::Align2::LEFT_TOP, placeholder, font.clone(), WEAK);
            }
        } else {
            clip.galley(origin, galley.clone(), TEXT);
        }
        let row_h = p.fonts(|f| f.row_height(&font));
        let caret_rect = Rect::from_min_size(origin + caret.min.to_vec2(), Vec2::new(2.0 * s, caret.height().max(row_h)));
        clip.rect_filled(caret_rect, 1.0, ACCENT_LIGHT);
        // Counter.
        let count = format!("{} / {}", snap.text.len(), snap.max_text_length);
        let full = snap.text.len() as u32 >= snap.max_text_length;
        p.text(
            Pos2::new(field.max.x, field.max.y + 4.0 * s),
            egui::Align2::RIGHT_TOP,
            count,
            FontId::proportional(13.0 * s),
            if full || self.refused { WARN } else { WEAK },
        );
        if self.shift != ShiftState::Off && self.page != Page::Number {
            let what = if self.shift == ShiftState::Locked { "CAPS" } else { "Shift" };
            p.text(Pos2::new(field.min.x, field.max.y + 4.0 * s), egui::Align2::LEFT_TOP, what, FontId::proportional(13.0 * s), ACCENT_LIGHT);
        }
        // Keys.
        let mut clicked = None;
        if snap.key_panel_visible {
            let sel = self.sel.fit(&self.layout);
            for (row, keys) in self.layout.rows.iter().enumerate() {
                let mut col = 0u8;
                for (index, key) in keys.iter().enumerate() {
                    let rect = geo.key_rect(row, col, key.span);
                    let here = Selection { row, index, column: col };
                    col += key.span;
                    let id = ui.id().with(("ime-key", row, index));
                    let r = ui.interact(rect, id, Sense::CLICK);
                    let selected = sel.row == row && sel.index == index;
                    let active = matches!(key.action, KeyAction::Shift if self.shift != ShiftState::Off)
                        || matches!(key.action, KeyAction::Symbols if self.page == Page::Symbols);
                    let (fill, color) = if selected {
                        (KEY_SELECTED, KEY_SELECTED_TEXT)
                    } else if key.action == KeyAction::Accept {
                        (if r.hovered() { ACCENT_LIGHT } else { ACCENT }, TEXT)
                    } else if r.hovered() {
                        (KEY_HOVER, TEXT)
                    } else {
                        (KEY, TEXT)
                    };
                    p.rect_filled(rect, 6.0 * s, fill);
                    if active {
                        p.rect_stroke(rect, 6.0 * s, Stroke::new(2.0f32, ACCENT_LIGHT), egui::StrokeKind::Inside);
                    }
                    let label = self.label(key.action, snap);
                    let size = if matches!(key.action, KeyAction::Char(_)) { 19.0 } else { 15.0 };
                    p.text(rect.center(), egui::Align2::CENTER_CENTER, label, FontId::proportional(size * s), color);
                    if r.clicked() {
                        clicked = Some((here, key.action));
                    }
                }
            }
        }
        // Hints.
        let hints = if self.page == Page::Number {
            "Cross: key   Circle: cancel\nSquare: delete   R2: done\nKeyboard: type, Enter done, Esc cancel"
        } else {
            "Cross: key   Circle: cancel   Triangle: space   Square: delete\nL1 / R1: move   L2: shift   R2: done   Keyboard: type, Enter done, Esc cancel"
        };
        let g = p.layout(hints.to_string(), FontId::proportional(12.0 * s), WEAK, inner.width());
        p.galley(Pos2::new(inner.min.x, geo.hints_top), g, WEAK);
        clicked
    }

    fn label(&self, action: KeyAction, snap: &HostSnapshot) -> String {
        match action {
            KeyAction::Char(c) => self.typed(c).to_string(),
            KeyAction::Shift => "Shift".into(),
            KeyAction::Symbols => if self.page == Page::Symbols { "ABC" } else { "#+=" }.into(),
            KeyAction::Space => "Space".into(),
            KeyAction::Backspace => "Delete".into(),
            KeyAction::CaretLeft => "<".into(),
            KeyAction::CaretRight => ">".into(),
            KeyAction::Accept => enter_label(snap.enter_label).into(),
            KeyAction::Cancel => "Cancel".into(),
        }
    }
}

/// The accept key's label for SceImeEnterLabel.
pub fn enter_label(label: u32) -> &'static str {
    match label {
        1 => "Send",
        2 => "Search",
        3 => "Go",
        _ => "OK",
    }
}

const PANEL: Color32 = Color32::from_rgba_premultiplied(26, 28, 34, 248);
const PANEL_EDGE: Color32 = Color32::from_rgb(58, 62, 74);
const FIELD: Color32 = Color32::from_rgb(14, 15, 19);
const KEY: Color32 = Color32::from_rgb(50, 54, 64);
const KEY_HOVER: Color32 = Color32::from_rgb(70, 75, 88);
const KEY_SELECTED: Color32 = Color32::from_rgb(236, 238, 242);
const KEY_SELECTED_TEXT: Color32 = Color32::from_rgb(20, 22, 28);
const ACCENT: Color32 = Color32::from_rgb(0, 102, 204);
const ACCENT_LIGHT: Color32 = Color32::from_rgb(64, 156, 255);
const TEXT: Color32 = Color32::from_rgb(240, 242, 246);
const WEAK: Color32 = Color32::from_rgb(150, 156, 168);
const WARN: Color32 = Color32::from_rgb(255, 150, 90);

/// Where the parts of the panel go on this screen.
struct Geometry {
    panel: Rect,
    field: Rect,
    grid_top: f32,
    hints_top: f32,
    key: Vec2,
    gap: f32,
    pad: f32,
    scale: f32,
}

impl Geometry {
    fn new(screen: Rect, snap: &HostSnapshot, layout: &Layout) -> Self {
        let number = layout.columns == 3;
        // About the console's share of the screen, readable from a couch.
        let width = if number { (screen.width() * 0.26).clamp(240.0, 380.0) } else { (screen.width() * 0.5).clamp(340.0, 760.0) };
        let scale = (width / if number { 300.0 } else { 600.0 }).clamp(0.75, 1.3);
        let pad = 18.0 * scale;
        let gap = 6.0 * scale;
        let columns = layout.columns as f32;
        let key_w = (width - 2.0 * pad - (columns - 1.0) * gap) / columns;
        let key = Vec2::new(key_w, (key_w * if number { 0.5 } else { 0.78 }).clamp(28.0, 50.0 * scale));
        let lines = if snap.option & OPTION_MULTILINE != 0 { 3.0 } else { 1.0 };
        let field_h = 20.0 * scale * 1.3 * lines + 18.0 * scale;
        let title_h = 30.0 * scale;
        let counter_h = 24.0 * scale;
        let rows = if snap.key_panel_visible { layout.rows.len() as f32 } else { 0.0 };
        let grid_h = if rows > 0.0 { rows * key.y + (rows - 1.0) * gap + 10.0 * scale } else { 0.0 };
        let hint_lines = if number { 3.0 } else { 2.0 };
        let hints_h = hint_lines * 15.0 * scale + 4.0 * scale;
        let height = pad + title_h + field_h + counter_h + grid_h + hints_h + pad;
        // The requested position (1920x1080 or 3840x2160 space) and
        // alignment, kept on the screen.
        let (bw, bh) = if snap.option & OPTION_USE_OVER_2K != 0 { (3840.0, 2160.0) } else { (1920.0, 1080.0) };
        let anchor = Pos2::new(screen.min.x + snap.posx / bw * screen.width(), screen.min.y + snap.posy / bh * screen.height());
        let frac = |a: u32| match a {
            1 => 0.5,
            2 => 1.0,
            _ => 0.0,
        };
        let margin = 8.0;
        let x = (anchor.x - width * frac(snap.horizontal_alignment)).clamp(screen.min.x + margin, (screen.max.x - width - margin).max(screen.min.x + margin));
        let y = (anchor.y - height * frac(snap.vertical_alignment)).clamp(screen.min.y + margin, (screen.max.y - height - margin).max(screen.min.y + margin));
        let panel = Rect::from_min_size(Pos2::new(x, y), Vec2::new(width, height));
        let field = Rect::from_min_size(Pos2::new(x + pad, y + pad + title_h), Vec2::new(width - 2.0 * pad, field_h));
        let grid_top = field.max.y + counter_h;
        let hints_top = grid_top + grid_h;
        Geometry { panel, field, grid_top, hints_top, key, gap, pad, scale }
    }

    fn key_rect(&self, row: usize, column: u8, span: u8) -> Rect {
        let x = self.panel.min.x + self.pad + column as f32 * (self.key.x + self.gap);
        let y = self.grid_top + row as f32 * (self.key.y + self.gap);
        let w = span as f32 * self.key.x + (span as f32 - 1.0) * self.gap;
        Rect::from_min_size(Pos2::new(x, y), Vec2::new(w, self.key.y))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ps5hle::ime_common::{utf16, TextEditEngine, TYPE_BASIC_LATIN};

    /// A dialog session over the HLE text engine (same filtering).
    struct Fake {
        generation: u64,
        editor: TextEditEngine,
        ended: Option<bool>,
        snap: HostSnapshot,
    }

    impl Fake {
        fn new(ty: u32, option: u32, max: u32, text: &str) -> Self {
            let mut editor = TextEditEngine::new();
            editor.reset(ty, option, max, utf16(text));
            let snap = HostSnapshot {
                generation: 1,
                ty,
                enter_label: 0,
                option,
                max_text_length: max,
                cursor: 0,
                disable_device: 0,
                key_panel_visible: true,
                posx: 960.0,
                posy: 540.0,
                horizontal_alignment: 1,
                vertical_alignment: 1,
                panel_width: 793,
                panel_height: 528,
                text: Vec::new(),
                title: utf16("Name your character"),
                placeholder: utf16("Name"),
            };
            Fake { generation: 1, editor, ended: None, snap }
        }

        fn snapshot(&self) -> HostSnapshot {
            HostSnapshot { text: self.editor.text().to_vec(), cursor: self.editor.cursor(), ..self.snap.clone() }
        }

        fn text(&self) -> String {
            String::from_utf16_lossy(self.editor.text())
        }
    }

    impl Session for Fake {
        fn insert(&mut self, g: u64, text: &[u16]) -> bool {
            g == self.generation && self.ended.is_none() && self.editor.insert(text).is_some()
        }
        fn backspace(&mut self, g: u64) -> bool {
            g == self.generation && self.editor.backspace().is_some()
        }
        fn move_cursor(&mut self, g: u64, d: i32) -> bool {
            g == self.generation && self.editor.move_cursor(d)
        }
        fn accept(&mut self, _: u64) -> bool {
            self.ended = Some(true);
            true
        }
        fn cancel(&mut self, _: u64) -> bool {
            self.ended = Some(false);
            true
        }
    }

    #[test]
    fn grid_moves_wrap_and_keep_the_column() {
        let l = Layout::for_page(Page::Letters);
        assert!(l.rows.iter().all(|r| r.iter().map(|k| k.span).sum::<u8>() == l.columns), "every row fills the grid");
        let start = Selection { row: 1, index: 6, column: 6 };
        assert_eq!(l.key(start).action, KeyAction::Char('u'));
        // Down through the wide Space key and back up to the same column.
        let down = start.step(&l, Dir::Down).step(&l, Dir::Down).step(&l, Dir::Down);
        assert_eq!(l.key(down).action, KeyAction::Space);
        let up = down.step(&l, Dir::Up).step(&l, Dir::Up).step(&l, Dir::Up);
        assert_eq!(l.key(up).action, KeyAction::Char('u'));
        // Left / right wrap in the row; right from Space starts its column
        // at Delete.
        let first = Selection { row: 1, index: 0, column: 0 };
        assert_eq!(l.key(first.step(&l, Dir::Left)).action, KeyAction::Char('p'));
        let del = down.step(&l, Dir::Right);
        assert_eq!((l.key(del).action, del.column), (KeyAction::Backspace, 8));
        assert_eq!(l.key(del.step(&l, Dir::Down)).action, KeyAction::Accept);
        // Up from the top row wraps to the bottom one.
        let top = Selection { row: 0, index: 0, column: 0 };
        assert_eq!(l.key(top.step(&l, Dir::Up)).action, KeyAction::CaretLeft);
        // The number pad.
        let n = Layout::for_page(Page::Number);
        assert!(n.rows.iter().all(|r| r.iter().map(|k| k.span).sum::<u8>() == n.columns));
        let ok = n.find(KeyAction::Accept).unwrap();
        assert_eq!(ok, Selection { row: 5, index: 1, column: 1 });
        assert_eq!(n.key(ok.step(&n, Dir::Up)).action, KeyAction::Backspace);
        assert_eq!(n.key(Selection { row: 5, index: 1, column: 2 }.step(&n, Dir::Down)).action, KeyAction::Char('3'));
    }

    #[test]
    fn pad_and_keyboard_mapping() {
        assert_eq!(nav_command(Nav::Accept), Some(Command::Press));
        assert_eq!(nav_command(Nav::Back), Some(Command::Cancel));
        assert_eq!(nav_command(Nav::Triangle), Some(Command::Space));
        assert_eq!(nav_command(Nav::Square), Some(Command::Backspace));
        assert_eq!(nav_command(Nav::PrevTab), Some(Command::Caret(-1)));
        assert_eq!(nav_command(Nav::NextTab), Some(Command::Caret(1)));
        assert_eq!(nav_command(Nav::L2), Some(Command::Shift));
        assert_eq!(nav_command(Nav::R2), Some(Command::Accept));
        assert_eq!(nav_command(Nav::Close), Some(Command::Accept));
        let key = |key, modifiers| egui::Event::Key { key, physical_key: Some(key), pressed: true, repeat: false, modifiers };
        let events = vec![
            key(egui::Key::A, egui::Modifiers::NONE),
            egui::Event::Text("a".into()),
            key(egui::Key::Backspace, egui::Modifiers::NONE),
            key(egui::Key::ArrowLeft, egui::Modifiers::NONE),
            key(egui::Key::ArrowRight, egui::Modifiers::NONE),
            key(egui::Key::Enter, egui::Modifiers::NONE),
            key(egui::Key::Escape, egui::Modifiers::NONE),
        ];
        let single = keyboard_commands(&events, false);
        assert_eq!(
            single,
            [Command::Insert("a".into()), Command::Backspace, Command::Caret(-1), Command::Caret(1), Command::Accept, Command::Cancel]
        );
        // Multi-line: Enter breaks the line, Ctrl+Enter accepts.
        let multi = keyboard_commands(&[key(egui::Key::Enter, egui::Modifiers::NONE), key(egui::Key::Enter, egui::Modifiers::CTRL)], true);
        assert_eq!(multi, [Command::Insert("\n".into()), Command::Accept]);
    }

    #[test]
    fn keys_type_through_the_session() {
        let mut f = Fake::new(TYPE_DEFAULT, 0, 6, "");
        let mut ui = ImeDialogUi::default();
        ui.begin(&f.snapshot());
        assert_eq!(ui.shift, ShiftState::Once, "an empty name starts with a capital");
        let press = |ui: &mut ImeDialogUi, f: &mut Fake, c: Command| ui.apply(c, f, 1);
        // Row 1 starts at 'q'; right twice is 'e'.
        assert_eq!(ui.layout().key(ui.sel).action, KeyAction::Char('q'));
        press(&mut ui, &mut f, Command::Press);
        press(&mut ui, &mut f, Command::Move(Dir::Right));
        press(&mut ui, &mut f, Command::Move(Dir::Right));
        press(&mut ui, &mut f, Command::Press);
        assert_eq!(f.text(), "Qe", "shift is used up by the first letter");
        press(&mut ui, &mut f, Command::Space);
        press(&mut ui, &mut f, Command::Insert("xyz".into()));
        assert_eq!(f.text(), "Qe xyz");
        // Full: the next key is refused.
        press(&mut ui, &mut f, Command::Press);
        assert!(ui.refused && f.text() == "Qe xyz");
        press(&mut ui, &mut f, Command::Backspace);
        press(&mut ui, &mut f, Command::Caret(-1));
        press(&mut ui, &mut f, Command::Shift);
        press(&mut ui, &mut f, Command::Shift);
        assert_eq!(ui.shift, ShiftState::Locked);
        press(&mut ui, &mut f, Command::Press);
        assert_eq!(f.text(), "Qe xEy");
        assert_eq!(ui.shift, ShiftState::Locked, "a locked shift stays");
        press(&mut ui, &mut f, Command::Accept);
        assert_eq!(f.ended, Some(true));

        // A number field takes digits only, whatever is typed.
        let mut f = Fake::new(TYPE_NUMBER, 0, 8, "");
        let mut ui = ImeDialogUi::default();
        ui.begin(&f.snapshot());
        assert_eq!(ui.page, Page::Number);
        ui.apply(Command::Insert("a1b2".into()), &mut f, 1);
        ui.apply(Command::Press, &mut f, 1);
        assert_eq!(f.text(), "124");
        ui.apply(Command::Cancel, &mut f, 1);
        assert_eq!(f.ended, Some(false));

        // Symbols page and back; basic latin refuses nothing printable.
        let mut f = Fake::new(TYPE_BASIC_LATIN, 0, 8, "");
        let mut ui = ImeDialogUi::default();
        ui.begin(&f.snapshot());
        assert_eq!(ui.shift, ShiftState::Off);
        ui.press(KeyAction::Symbols, &mut f, 1);
        assert_eq!(ui.page, Page::Symbols);
        ui.apply(Command::Press, &mut f, 1);
        assert_eq!(f.text(), "!");
        ui.press(KeyAction::Symbols, &mut f, 1);
        assert_eq!(ui.page, Page::Letters);
    }

    #[test]
    fn unattended_runs_do_not_wait() {
        let vars = |pairs: &'static [(&'static str, &'static str)]| move |k: &str| pairs.iter().find(|(n, _)| *n == k).map(|(_, v)| v.to_string());
        assert!(waits_for_user(vars(&[])));
        assert!(waits_for_user(vars(&[(HEADLESS_ENV, "0"), ("RPS5E_NO_HOST_INPUT", "0")])));
        assert!(!waits_for_user(vars(&[(HEADLESS_ENV, "1")])));
        assert!(!waits_for_user(vars(&[("RPS5E_NO_HOST_INPUT", "1")])));
        assert!(!waits_for_user(vars(&[("RPS5E_PAD_SCRIPT", "10:cross")])));
        assert!(!waits_for_user(vars(&[("RPS5E_INPUT_REPLAY", "rec.txt")])));
        assert!(!waits_for_user(vars(&[("RPS5E_INPUT_RECORD", "rec.txt")])));
        assert!(waits_for_user(vars(&[(HEADLESS_ENV, "0"), ("RPS5E_PAD_SCRIPT", "10:cross")])));
    }

    #[test]
    fn clicked_key_presses_and_draws() {
        let mut f = Fake::new(TYPE_DEFAULT, 0, 16, "");
        let mut ui = ImeDialogUi::default();
        let ctx = egui::Context::default();
        let screen = Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 720.0));
        let run = |ui: &mut ImeDialogUi, f: &mut Fake, events: Vec<egui::Event>| {
            let raw = egui::RawInput { screen_rect: Some(screen), events, ..Default::default() };
            ctx.run(raw, |c| {
                let snap = f.snapshot();
                ui.frame_input(c, &snap, f);
                let snap = f.snapshot();
                ui.draw(c, &snap, f);
            })
        };
        run(&mut ui, &mut f, Vec::new());
        let out = run(&mut ui, &mut f, Vec::new());
        assert!(!out.shapes.is_empty());
        // Click the 'w' key (row 1, index 1).
        let geo = Geometry::new(screen, &f.snapshot(), ui.layout());
        let at = geo.key_rect(1, 1, 1).center();
        let click = |pressed| egui::Event::PointerButton { pos: at, button: egui::PointerButton::Primary, pressed, modifiers: egui::Modifiers::NONE };
        run(&mut ui, &mut f, vec![egui::Event::PointerMoved(at)]);
        run(&mut ui, &mut f, vec![click(true)]);
        run(&mut ui, &mut f, vec![click(false)]);
        assert_eq!(f.text(), "W");
        assert_eq!(ui.sel, Selection { row: 1, index: 1, column: 1 });
        // The physical keyboard and queued pad steps.
        ui.queue_nav(&[Nav::Triangle]);
        run(&mut ui, &mut f, vec![egui::Event::Text("ow".into())]);
        assert_eq!(f.text(), "Wow ");
        run(&mut ui, &mut f, vec![egui::Event::Key { key: egui::Key::Enter, physical_key: None, pressed: true, repeat: false, modifiers: egui::Modifiers::NONE }]);
        assert_eq!(f.ended, Some(true));
    }

    /// `RPS5E_GUI_RENDER_DIR=<dir> cargo test -p rps5e-gui render_ime -- --ignored`
    /// writes the keyboard over a dark game frame, a password field and the
    /// number pad as PNG files.
    #[test]
    #[ignore]
    fn render_ime_dialog() {
        let Some(out) = std::env::var_os("RPS5E_GUI_RENDER_DIR") else { return };
        let out = std::path::PathBuf::from(out);
        std::fs::create_dir_all(&out).unwrap();
        let shots: [(&str, u32, u32, &str, i32); 3] = [
            ("ime-dialog.png", TYPE_DEFAULT, 0, "Boletaria", -2),
            ("ime-dialog-password.png", TYPE_BASIC_LATIN, OPTION_PASSWORD, "secret", 0),
            ("ime-dialog-number.png", TYPE_NUMBER, 0, "", 0),
        ];
        for (name, ty, option, text, caret) in shots {
            let mut f = Fake::new(ty, option, 16, text);
            f.editor.move_cursor(caret);
            if ty == TYPE_NUMBER {
                f.snap.title = utf16("Enter the amount");
                f.snap.placeholder = utf16("0");
            }
            let mut ui = ImeDialogUi::default();
            ui.begin(&f.snapshot());
            ui.apply(Command::Move(Dir::Right), &mut f, 1);
            ui.apply(Command::Move(Dir::Down), &mut f, 1);
            let mut r = crate::softrender::SoftRenderer::new([1280.0, 720.0], 1.0);
            let mut img = (0, 0, Vec::new());
            for _ in 0..20 {
                let snap = f.snapshot();
                img = r.frame(Vec::new(), |ctx| {
                    // A stand-in game frame behind the dialog.
                    egui::Area::new(egui::Id::new("game")).order(egui::Order::Background).fixed_pos(Pos2::ZERO).show(ctx, |ui| {
                        let screen = ctx.screen_rect();
                        ui.painter().rect_filled(screen, 0.0, Color32::from_rgb(60, 70, 64));
                        ui.painter().rect_filled(Rect::from_min_size(Pos2::new(80.0, 80.0), Vec2::new(500.0, 300.0)), 0.0, Color32::from_rgb(120, 96, 70));
                    });
                    ui.draw(ctx, &snap, &mut f);
                });
            }
            crate::softrender::write_png(&out.join(name), img.0, img.1, &img.2);
        }
    }
}
