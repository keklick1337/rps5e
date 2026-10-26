// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Sections of the library (the library folder a game was found in, or
//! the games added one by one) and removal of a game or a whole section
//! from the library. Removal only changes the launcher's settings: the
//! games stay on disk, and a snapshot of the library settings taken
//! before the removal undoes it.

use crate::library::Game;
use crate::settings::{GuiSettings, HiddenGame, LibraryFolder};
use std::path::{Path, PathBuf};

/// A group of the library list.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Section {
    /// A library folder, searched for games.
    Folder(PathBuf),
    /// Games added one by one (Add game...).
    Added,
}

impl Section {
    /// Header text: the folder's path (two folders may share a name), or
    /// "Added games".
    pub fn label(&self) -> String {
        match self {
            Section::Folder(p) => p.display().to_string(),
            Section::Added => "Added games".into(),
        }
    }

    /// Hover text: the folder's path.
    pub fn detail(&self) -> String {
        match self {
            Section::Folder(p) => format!("Library folder {}", p.display()),
            Section::Added => "Games added one by one (Add game...)".into(),
        }
    }
}

/// Whether `entry` (an `extra_games` path) is `game`.
fn is_entry_of(entry: &Path, game: &Game) -> bool {
    entry == game.source.path() || entry == game.dir
}

/// The section `game` is listed under: the games added by hand first (a
/// removed library folder keeps them), else the innermost library folder
/// holding it.
pub fn section_of(settings: &GuiSettings, roots: &[PathBuf], game: &Game) -> Section {
    if settings.extra_games.iter().any(|e| is_entry_of(e, game)) {
        return Section::Added;
    }
    let path = game.source.path();
    roots
        .iter()
        .filter(|r| path.starts_with(r))
        .max_by_key(|r| r.components().count())
        .map(|r| Section::Folder(r.clone()))
        .unwrap_or(Section::Added)
}

/// `games` grouped by section, in the order of the library folders, the
/// added games last; sections without games are left out.
pub fn group(settings: &GuiSettings, games: &[Game]) -> Vec<(Section, Vec<Game>)> {
    let roots = settings.library_roots();
    let mut out: Vec<(Section, Vec<Game>)> = roots.iter().map(|r| (Section::Folder(r.clone()), Vec::new())).collect();
    out.push((Section::Added, Vec::new()));
    for g in games {
        let s = section_of(settings, &roots, g);
        if let Some((_, v)) = out.iter_mut().find(|(k, _)| *k == s) {
            v.push(g.clone());
        }
    }
    out.retain(|(_, v)| !v.is_empty());
    out
}

/// What to remove from the library.
#[derive(Clone, Debug, PartialEq)]
pub enum Removal {
    /// One game (every copy with its title id).
    Game(Game),
    /// A library folder, or every game added by hand.
    Section(Section),
}

/// The confirmation of a removal.
#[derive(Clone, Debug, PartialEq)]
pub struct RemovalPlan {
    pub title: String,
    pub text: String,
    /// Keys of the listed games it removes.
    pub keys: Vec<String>,
}

fn plural(n: usize) -> &'static str {
    if n == 1 {
        "game"
    } else {
        "games"
    }
}

const KEPT_ON_DISK: &str = "Game files on disk are not deleted.";

/// What removing `r` from `games` (the listed games) does.
pub fn plan(settings: &GuiSettings, games: &[Game], r: &Removal) -> RemovalPlan {
    match r {
        Removal::Game(g) => {
            let h = HiddenGame::of(g);
            let keys: Vec<String> = games.iter().filter(|x| h.matches(x)).map(Game::key).collect();
            let id = g.title_id.as_deref().map(|id| format!(" ({id})")).unwrap_or_default();
            let copies = match keys.len() {
                0 | 1 => String::new(),
                n => format!("\nAll {n} copies with this title id are removed."),
            };
            RemovalPlan {
                title: "Remove from library".into(),
                text: format!(
                    "Remove \"{}\"{id} from the library?{copies}\n\n{KEPT_ON_DISK}\nUndo (status bar) or Settings > Library > Hidden games brings it back.",
                    g.title
                ),
                keys,
            }
        }
        Removal::Section(s) => {
            let roots = settings.library_roots();
            let keys: Vec<String> = games.iter().filter(|g| section_of(settings, &roots, g) == *s).map(Game::key).collect();
            let n = keys.len();
            let text = match s {
                Section::Folder(p) => format!(
                    "Remove the library folder\n{}\nand its {n} {} from the library?\n\nThe folder is no longer searched for games. {KEPT_ON_DISK}\nTo keep it in the list without scanning it, turn it off in Settings > Library instead.\nUndo (status bar) or Add > Library folder brings it back.",
                    p.display(),
                    plural(n)
                ),
                Section::Added => format!(
                    "Remove the {n} {} added one by one from the library?\n\n{KEPT_ON_DISK}\nUndo (status bar) or Add > Game brings them back.",
                    plural(n)
                ),
            };
            let title = match s {
                Section::Folder(_) => "Remove library folder",
                Section::Added => "Remove added games",
            };
            RemovalPlan { title: title.into(), text, keys }
        }
    }
}

/// The library settings a removal changes.
#[derive(Clone, Debug, PartialEq)]
pub struct LibrarySnapshot {
    default_library: bool,
    library_folders: Vec<LibraryFolder>,
    extra_games: Vec<PathBuf>,
    hidden_games: Vec<HiddenGame>,
}

impl LibrarySnapshot {
    pub fn of(s: &GuiSettings) -> Self {
        Self {
            default_library: s.default_library,
            library_folders: s.library_folders.clone(),
            extra_games: s.extra_games.clone(),
            hidden_games: s.hidden_games.clone(),
        }
    }

    /// Put the library settings back.
    pub fn restore(self, s: &mut GuiSettings) {
        s.default_library = self.default_library;
        s.library_folders = self.library_folders;
        s.extra_games = self.extra_games;
        s.hidden_games = self.hidden_games;
    }
}

/// Remove `r` from the library; the snapshot undoes it. `games`: the
/// listed games.
pub fn apply(settings: &mut GuiSettings, games: &[Game], r: &Removal) -> LibrarySnapshot {
    let before = LibrarySnapshot::of(settings);
    match r {
        Removal::Game(g) => {
            settings.hide(g);
        }
        Removal::Section(Section::Folder(dir)) => {
            settings.remove_library_dir(dir);
        }
        Removal::Section(Section::Added) => {
            let roots = settings.library_roots();
            let added: Vec<&Game> = games.iter().filter(|g| section_of(settings, &roots, g) == Section::Added).collect();
            settings.extra_games.retain(|e| !added.iter().any(|g| is_entry_of(e, g)));
        }
    }
    before
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::library::GameSource;

    fn game(path: &str, id: Option<&str>) -> Game {
        Game {
            source: GameSource::Folder(path.into()),
            dir: path.into(),
            title: format!("T {path}"),
            title_id: id.map(str::to_string),
            ..Default::default()
        }
    }

    fn package(path: &str) -> Game {
        Game {
            source: GameSource::Package(path.into()),
            dir: Path::new(path).parent().unwrap().into(),
            title: format!("P {path}"),
            ..Default::default()
        }
    }

    fn settings() -> GuiSettings {
        let mut s = GuiSettings::default();
        s.default_library = false;
        s.library_folders = ["/lib", "/lib/nested", "/other"].into_iter().map(LibraryFolder::new).collect();
        s.extra_games = vec!["/home/u/single".into(), "/lib/hand-added".into(), "/pkgs/x.pkg".into()];
        s
    }

    fn library() -> Vec<Game> {
        vec![
            game("/lib/a", Some("PPSA00001")),
            game("/lib/nested/b", Some("PPSA00002")),
            game("/lib/hand-added", Some("PPSA00003")),
            game("/other/c", None),
            game("/home/u/single", Some("PPSA00004")),
            package("/pkgs/x.pkg"),
            game("/lib/d", None),
        ]
    }

    fn keys(v: &[Game]) -> Vec<String> {
        v.iter().map(Game::key).collect()
    }

    #[test]
    fn games_are_grouped_by_library_folder_then_added() {
        let s = settings();
        let g = group(&s, &library());
        let names: Vec<(Section, Vec<String>)> = g.into_iter().map(|(s, v)| (s, keys(&v))).collect();
        assert_eq!(
            names,
            [
                (Section::Folder("/lib".into()), vec!["/lib/a".to_string(), "/lib/d".into()]),
                (Section::Folder("/lib/nested".into()), vec!["/lib/nested/b".into()]),
                (Section::Folder("/other".into()), vec!["/other/c".into()]),
                (Section::Added, vec!["/lib/hand-added".into(), "/home/u/single".into(), "/pkgs/x.pkg".into()]),
            ]
        );
        assert_eq!(Section::Folder("/srv/games".into()).label(), "/srv/games");
        assert_eq!(Section::Added.label(), "Added games");
        // A folder without games has no section.
        let mut s = settings();
        s.library_folders.push(LibraryFolder::new("/empty"));
        assert!(group(&s, &library()).iter().all(|(s, _)| *s != Section::Folder("/empty".into())));
    }

    #[test]
    fn removing_a_folder_counts_its_games_and_undoes() {
        let mut s = settings();
        let games = library();
        let r = Removal::Section(Section::Folder("/lib".into()));
        let p = plan(&s, &games, &r);
        assert_eq!(p.keys, ["/lib/a", "/lib/d"], "the nested folder and the added game stay");
        assert!(p.text.contains("its 2 games") && p.text.contains("not deleted"), "{}", p.text);
        let before = s.clone();
        let undo = apply(&mut s, &games, &r);
        assert_eq!(s.library_roots(), [PathBuf::from("/lib/nested"), "/other".into()]);
        assert_eq!(s.extra_games, before.extra_games);
        // Files are not touched: only the settings change, and Undo puts
        // them back.
        undo.restore(&mut s);
        assert_eq!(s, before);
    }

    #[test]
    fn removing_added_games_drops_their_entries_only() {
        let mut s = settings();
        s.extra_games.push("/gone/not-found".into());
        let games = library();
        let r = Removal::Section(Section::Added);
        let p = plan(&s, &games, &r);
        assert_eq!(p.keys.len(), 3);
        assert!(p.text.contains("3 games added one by one"), "{}", p.text);
        let before = s.clone();
        let undo = apply(&mut s, &games, &r);
        assert_eq!(s.extra_games, [PathBuf::from("/gone/not-found")], "entries not in the list are kept");
        assert_eq!(s.library_folders, before.library_folders);
        undo.restore(&mut s);
        assert_eq!(s, before);
    }

    #[test]
    fn removing_one_game_hides_every_copy() {
        let mut s = settings();
        let mut games = library();
        games.push(game("/other/copy-of-a", Some("ppsa00001")));
        let r = Removal::Game(games[0].clone());
        let p = plan(&s, &games, &r);
        assert_eq!(p.keys, ["/lib/a", "/other/copy-of-a"]);
        assert!(p.text.contains("\"T /lib/a\" (PPSA00001)") && p.text.contains("All 2 copies"), "{}", p.text);
        let before = s.clone();
        let undo = apply(&mut s, &games, &r);
        assert!(s.is_hidden(&games[0]) && s.is_hidden(&games[7]) && !s.is_hidden(&games[1]));
        assert_eq!(s.library_folders, before.library_folders);
        undo.restore(&mut s);
        assert_eq!(s, before);
        // A game without a title id: that path only.
        let p = plan(&s, &games, &Removal::Game(games[3].clone()));
        assert_eq!(p.keys, ["/other/c"]);
        assert!(!p.text.contains("copies"));
    }
}
