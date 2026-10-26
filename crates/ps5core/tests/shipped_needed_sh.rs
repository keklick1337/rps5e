// RustPS5Emu (RPS5E)
// Copyright (C) 2025-2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Loader check on SILENT HILL (SILENT HILL PPSA10112 under `GAMES_DIR`: the game folder; skipped when unset
//! or absent): the module the eboot lists as needed, `libSceNpCppWebApi`
//! shipped in `sce_module`, is loaded after libc and before linking, the
//! eboot's imports bind to its exports, and every module comes after the
//! modules it needs. Only loads; no guest code runs.

fn game() -> Option<String> {
    let dir = ps5vm::test_games::game_dir("PPSA10112")?.to_string_lossy().into_owned();
    let eboot = format!("{dir}/eboot.bin");
    std::path::Path::new(&eboot).is_file().then_some(eboot)
}

#[test]
fn needed_module_is_loaded_before_linking() {
    let Some(eboot) = game() else {
        eprintln!("game not present, skipped");
        return;
    };
    let game = ps5core::load_eboot(&[], &eboot).expect("load");
    let names: Vec<&str> = game.programs.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(names[..3], ["eboot.bin", "libc.prx", "libSceNpCppWebApi.prx"], "load order");
    // Function imports of the eboot that the module exports are linked to its
    // code, unless the HLE implements them (the HLE keeps priority).
    let npcpp = &game.programs[2];
    let d = game.programs[0].dynamic.as_ref().expect("eboot dynamic");
    let mut bound = 0;
    for r in d.relas.iter().chain(d.jmp_rels.iter()) {
        let Some(sym) = d.symbol(r.sym()) else { continue };
        if r.sym() == 0 || sym.st_shndx != 0 || sym.type_() != 2 {
            continue;
        }
        let nid = d.name(sym.st_name);
        let nid = nid.split('#').next().unwrap_or_default();
        if npcpp.resolve(nid).is_some() && !game.hle.is_implemented(nid) {
            assert!(game.lle_imports.contains_key(nid), "{nid} not linked to libSceNpCppWebApi");
            bound += 1;
        }
    }
    assert!(bound > 100, "imports linked to libSceNpCppWebApi: {bound}");
    for (i, p) in game.programs.iter().enumerate().skip(1) {
        for dep in ps5core::needed::needed_modules(p) {
            if let Some(j) = game.programs.iter().position(|q| q.name == dep) {
                assert!(j < i, "{} is loaded before its needed module {dep}", p.name);
            }
        }
    }
}
