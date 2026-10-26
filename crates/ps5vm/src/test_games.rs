// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Finds a game folder for the tests that read real games: the folder under
//! `GAMES_DIR` (up to three levels deep) whose `sce_sys/param.json` carries
//! the given title ID. Without `GAMES_DIR` or the game those tests skip.

use std::path::{Path, PathBuf};

/// The game folder (the one holding `eboot.bin`) of `title_id`, if present.
pub fn game_dir(title_id: &str) -> Option<PathBuf> {
    let root = PathBuf::from(std::env::var_os("GAMES_DIR")?);
    find(&root, title_id, 3)
}

fn find(dir: &Path, title_id: &str, depth: u32) -> Option<PathBuf> {
    if matches(dir, title_id) {
        return Some(dir.to_path_buf());
    }
    if depth == 0 {
        return None;
    }
    let mut subdirs: Vec<PathBuf> = std::fs::read_dir(dir).ok()?.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
    subdirs.sort();
    subdirs.into_iter().find_map(|d| find(&d, title_id, depth - 1))
}

fn matches(dir: &Path, title_id: &str) -> bool {
    dir.join("eboot.bin").is_file()
        && std::fs::read_to_string(dir.join("sce_sys/param.json")).is_ok_and(|json| json.contains(&format!("\"{title_id}\"")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_a_game_by_title_id_below_the_root() {
        let root = std::env::temp_dir().join(format!("rps5e-test-games-{}", std::process::id()));
        let game = root.join("[PS5] Some Game/PPSA00001");
        std::fs::create_dir_all(game.join("sce_sys")).unwrap();
        std::fs::write(game.join("eboot.bin"), b"").unwrap();
        std::fs::write(game.join("sce_sys/param.json"), br#"{ "titleId": "PPSA00001" }"#).unwrap();
        assert_eq!(find(&root, "PPSA00001", 3), Some(game));
        assert_eq!(find(&root, "PPSA00002", 3), None);
        std::fs::remove_dir_all(&root).unwrap();
    }
}
