// RustPS5Emu (RPS5E)
// Copyright (C) 2025-2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Loader check on a real game image (Terminator 2D PPSA25872 under `GAMES_DIR`: the Terminator 2D folder;
//! skipped when unset or absent):
//! the main executable and libc.prx are mapped, relocated and linked, the
//! dynamic metadata (library/module ids, exports) is read, functions bind
//! to import stubs and segments get their program-header protections.
//! Only loads; no guest code runs.

fn game() -> Option<String> {
    let dir = ps5vm::test_games::game_dir("PPSA25872")?.to_string_lossy().into_owned();
    let eboot = format!("{dir}/eboot.bin");
    std::path::Path::new(&eboot).is_file().then_some(eboot)
}

#[test]
fn loads_main_executable_and_libc() {
    let Some(eboot) = game() else {
        eprintln!("game not present, skipped");
        return;
    };
    let game = ps5core::load_eboot(&[], &eboot).expect("load");
    assert_eq!(game.image_base, 0x8_0000_0000);
    let (main, libc) = (&game.programs[0], &game.programs[1]);
    // Besides libc, only modules the eboot lists as needed are loaded at start-up.
    let needed = ps5core::needed::needed_modules(main);
    assert!(game.programs[2..].iter().all(|p| needed.contains(&p.name)), "unexpected start-up module");
    assert!(!main.is_shared && libc.is_shared);
    assert_eq!((main.handle, libc.handle), (1, 2));
    let d = main.dynamic.as_ref().unwrap();
    assert!(!d.import_libs.is_empty() && !d.import_modules.is_empty());
    let ld = libc.dynamic.as_ref().unwrap();
    assert!(!ld.export_libs.is_empty() && !ld.export_modules.is_empty());
    assert!(libc.exports.len() > 100, "libc exports {}", libc.exports.len());
    assert!(libc.init_func != 0);
    // Every function import has a stub; libc data is bound.
    assert!(game.import_stubs.count() > 100);
    eprintln!(
        "unresolved relocations: main {} libc {}; stubs {} lle {}",
        main.unresolved_relocations(),
        libc.unresolved_relocations(),
        game.import_stubs.count(),
        game.lle_imports.len()
    );
    // Code is executable and not writable; data is writable.
    let code = main.segments.iter().find(|s| s.prot.exec).unwrap();
    assert!(!code.prot.write && ps5vm::ranges::accessible(code.addr, 8, false) && !ps5vm::ranges::accessible(code.addr, 8, true));
    assert!(main.segments.iter().any(|s| s.prot.write && ps5vm::ranges::accessible(s.addr, 8, true)));
    // Kernel module registry: unwind info and dlsym through the registry.
    let m = ps5hle::modules::module_by_address(main.base + main.entry).unwrap();
    assert!(m.is_main && m.eh_frame_hdr != 0 && m.eh_frame != 0);
    assert!(ps5hle::modules::module_by_address(libc.base).is_some_and(|m| m.handle == 2));
    assert!(ps5hle::modules::resolve_symbol(2, "malloc").is_some());
}
