// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Compatibility database: a local JSON file (`compatibility_db.json` in
//! the configuration directory) with a status and a comment per title id:
//! `{"PPSA01234": {"status": "InGame", "comment": "..."}}`.

use crate::settings::write_atomic;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub const FILE_NAME: &str = "compatibility_db.json";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum GameStatus {
    #[default]
    Unknown,
    InGame,
    Logo,
    DoesntBoot,
    MainMenu,
}

impl GameStatus {
    pub const ALL: [GameStatus; 5] =
        [GameStatus::Unknown, GameStatus::InGame, GameStatus::MainMenu, GameStatus::Logo, GameStatus::DoesntBoot];

    pub fn from_text(text: &str) -> Self {
        match text.trim() {
            "InGame" | "In game" => GameStatus::InGame,
            "MainMenu" | "Main menu" => GameStatus::MainMenu,
            "Logo" => GameStatus::Logo,
            "DoesntBoot" | "Doesn't boot" => GameStatus::DoesntBoot,
            _ => GameStatus::Unknown,
        }
    }

    /// File form.
    pub fn as_text(self) -> &'static str {
        match self {
            GameStatus::InGame => "InGame",
            GameStatus::MainMenu => "MainMenu",
            GameStatus::Logo => "Logo",
            GameStatus::DoesntBoot => "DoesntBoot",
            GameStatus::Unknown => "Unknown",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            GameStatus::InGame => "In game",
            GameStatus::MainMenu => "Main menu",
            GameStatus::Logo => "Logo",
            GameStatus::DoesntBoot => "Doesn't boot",
            GameStatus::Unknown => "Unknown",
        }
    }

    /// Indicator colour (RGB).
    pub fn color(self) -> [u8; 3] {
        match self {
            GameStatus::InGame => [76, 175, 80],
            GameStatus::MainMenu => [139, 195, 74],
            GameStatus::Logo => [255, 193, 7],
            GameStatus::DoesntBoot => [244, 67, 54],
            GameStatus::Unknown => [158, 158, 158],
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Entry {
    pub status: GameStatus,
    pub comment: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct CompatibilityDb {
    pub entries: BTreeMap<String, Entry>,
    pub path: Option<PathBuf>,
}

fn title_key(id: &str) -> String {
    id.trim().to_ascii_uppercase()
}

impl CompatibilityDb {
    pub fn parse(text: &str) -> Result<BTreeMap<String, Entry>, String> {
        let doc: serde_json::Value =
            serde_json::from_str(text).map_err(|e| format!("Invalid compatibility JSON: {e}"))?;
        let root = doc.as_object().ok_or("Invalid compatibility JSON: not an object")?;
        let mut out = BTreeMap::new();
        for (k, v) in root {
            let key = title_key(k);
            let Some(obj) = v.as_object().filter(|o| !o.is_empty()) else { continue };
            if key.is_empty() {
                continue;
            }
            let text = |n: &str| obj.get(n).and_then(|x| x.as_str()).unwrap_or("").to_string();
            out.insert(key, Entry { status: GameStatus::from_text(&text("status")), comment: text("comment") });
        }
        Ok(out)
    }

    pub fn to_json(&self) -> String {
        let mut root = serde_json::Map::new();
        for (k, e) in &self.entries {
            root.insert(
                k.clone(),
                serde_json::json!({"status": e.status.as_text(), "comment": e.comment}),
            );
        }
        serde_json::to_string_pretty(&serde_json::Value::Object(root)).unwrap_or_default()
    }

    /// Load `<dir>/compatibility_db.json` (empty when missing).
    pub fn load(dir: &Path) -> Result<Self, String> {
        let path = dir.join(FILE_NAME);
        let entries = match std::fs::read_to_string(&path) {
            Ok(t) => Self::parse(&t)?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => BTreeMap::new(),
            Err(e) => return Err(format!("Could not open compatibility database: {e}")),
        };
        Ok(Self { entries, path: Some(path) })
    }

    pub fn find(&self, title_id: &str) -> Option<&Entry> {
        self.entries.get(&title_key(title_id))
    }

    pub fn set_status(&mut self, title_id: &str, status: GameStatus) -> Result<(), String> {
        let key = title_key(title_id);
        if key.is_empty() {
            return Ok(());
        }
        self.entries.entry(key).or_default().status = status;
        self.save()
    }

    pub fn set_comment(&mut self, title_id: &str, comment: &str) -> Result<(), String> {
        let key = title_key(title_id);
        if key.is_empty() {
            return Ok(());
        }
        self.entries.entry(key).or_default().comment = comment.to_string();
        self.save()
    }

    fn save(&self) -> Result<(), String> {
        match &self.path {
            Some(p) => write_atomic(p, &self.to_json()).map_err(|e| format!("Could not save compatibility database: {e}")),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_edit_save() {
        let m = CompatibilityDb::parse(
            r#"{"ppsa10112": {"status": "Main menu", "comment": "slow"}, "PPSA21567": {"status": "weird"}, "X": {}}"#,
        )
        .unwrap();
        assert_eq!(m["PPSA10112"], Entry { status: GameStatus::MainMenu, comment: "slow".into() });
        assert_eq!(m["PPSA21567"].status, GameStatus::Unknown);
        assert!(!m.contains_key("X"));
        assert!(CompatibilityDb::parse("[1]").is_err());

        let dir = std::env::temp_dir().join(format!("rps5e-gui-compat-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut db = CompatibilityDb::load(&dir).unwrap();
        assert!(db.entries.is_empty());
        db.set_status("ppsa00001", GameStatus::InGame).unwrap();
        db.set_comment("PPSA00001", "works").unwrap();
        db.set_status(" ", GameStatus::Logo).unwrap();
        let back = CompatibilityDb::load(&dir).unwrap();
        assert_eq!(back.find("ppsa00001"), Some(&Entry { status: GameStatus::InGame, comment: "works".into() }));
        assert_eq!(back.entries.len(), 1);
        for s in GameStatus::ALL {
            assert_eq!(GameStatus::from_text(s.as_text()), s);
            assert_eq!(GameStatus::from_text(s.label()), s);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
