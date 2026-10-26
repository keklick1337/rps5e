// RustPS5Emu (RPS5E)
// Copyright (C) 2025-2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Loader check on Marvel's Spider-Man 2 (Marvel's Spider-Man 2 PPSA08338 under `GAMES_DIR`: the game folder; skipped
//! when unset or absent): the modules the eboot lists as needed are loaded at
//! start-up, including the two kept in the game root
//! (`libcohtml.Prospero.prx`, `libRenoirCore.PS5.prx`), and every module comes
//! after the modules it needs. Only loads; no guest code runs.

fn game() -> Option<String> {
    let dir = ps5vm::test_games::game_dir("PPSA08338")?.to_string_lossy().into_owned();
    let eboot = format!("{dir}/eboot.bin");
    std::path::Path::new(&eboot).is_file().then_some(eboot)
}

#[test]
fn needed_modules_are_loaded_before_linking() {
    let Some(eboot) = game() else {
        eprintln!("game not present, skipped");
        return;
    };
    let game = ps5core::load_eboot(&[], &eboot).expect("load");
    let names: Vec<&str> = game.programs.iter().map(|p| p.name.as_str()).collect();
    for want in ["libSceNpCppWebApi.prx", "libcohtml.Prospero.prx", "libRenoirCore.PS5.prx"] {
        assert!(names.contains(&want), "{want} not loaded: {names:?}");
    }
    for (i, p) in game.programs.iter().enumerate().skip(1) {
        for dep in ps5core::needed::needed_modules(p) {
            if let Some(j) = game.programs.iter().position(|q| q.name == dep) {
                assert!(j < i, "{} is loaded before its needed module {dep}", p.name);
            }
        }
    }
}
