// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! `info-pkg` / `install-pkg` on synthetic packages.

use ps5pkg::build::{fake_pkg_image, PfsBuilder, PkgBuilder};
use ps5pkg::param::{build_sfo, SfoValue};
use std::path::{Path, PathBuf};
use std::process::Command;

const CID: &str = "UP9000-PPSA99999_00-TESTGAME00000000";

fn tree(version: &str, eboot: &[u8]) -> PfsBuilder {
    let json = format!(
        r#"{{"titleId":"PPSA99999","contentId":"{CID}","contentVersion":"{version}",
           "localizedParameters":{{"defaultLanguage":"en-US","en-US":{{"titleName":"CLI Test"}}}}}}"#
    );
    PfsBuilder::new()
        .file("eboot.bin", eboot)
        .file("sce_sys/param.json", json.as_bytes())
        .file("Data/a.bin", &[1u8; 3000])
}

fn pkg(dir: &Path, name: &str, version: &str, eboot: &[u8], patch: bool, passcode: &str) -> PathBuf {
    let mut b = PkgBuilder::new(CID, fake_pkg_image(&tree(version, eboot), CID, passcode, false));
    b.passcode = Some(passcode.to_string());
    b.param_sfo = Some(build_sfo(&[
        ("CATEGORY", SfoValue::Str(if patch { "gp" } else { "gd" }.into())),
        ("TITLE_ID", SfoValue::Str("PPSA99999".into())),
    ]));
    if patch {
        b.content_flags = ps5pkg::pkg::CONTENT_FLAG_FIRST_PATCH;
    }
    let p = dir.join(name);
    std::fs::write(&p, b.build()).unwrap();
    p
}

fn cli(args: &[&str]) -> (bool, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_rps5e-cli")).args(args).output().unwrap();
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    (out.status.success(), text)
}

#[test]
fn info_install_update_and_refusals() {
    let dir = std::env::temp_dir().join(format!("rps5e-cli-pkg-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let zeros = ps5pkg::crypto::DEFAULT_PASSCODE;
    let game = pkg(&dir, "game.pkg", "01.000.000", b"EBOOT v1", false, zeros);
    let s = |p: &Path| p.to_string_lossy().into_owned();

    let (ok, out) = cli(&["info-pkg", &s(&game)]);
    assert!(ok, "{out}");
    for want in ["PPSA99999", "CLI Test", "01.000.000", CID, "Status:      ready", "Data/a.bin", "3000"] {
        assert!(out.contains(want), "missing {want:?} in:\n{out}");
    }

    // Default destination: <data>/games/<TITLE_ID>.
    let data = dir.join("data");
    let (ok, out) = cli(&["install-pkg", &s(&game), "--data-dir", &s(&data)]);
    assert!(ok, "{out}");
    let dest = data.join("games/PPSA99999");
    assert_eq!(std::fs::read(dest.join("eboot.bin")).unwrap(), b"EBOOT v1");
    assert_eq!(std::fs::read(dest.join("Data/a.bin")).unwrap(), vec![1u8; 3000]);

    // An update goes over the installed game; it cannot run on its own.
    let patch = pkg(&dir, "patch.pkg", "01.001.000", b"EBOOT v2", true, zeros);
    let (ok, out) = cli(&["install-pkg", &s(&patch), "--into", &s(&dir.join("elsewhere"))]);
    assert!(!ok && out.contains("install the game first"), "{out}");
    let (ok, out) = cli(&["install-pkg", &s(&patch), "--data-dir", &s(&data)]);
    assert!(ok, "{out}");
    assert_eq!(std::fs::read(dest.join("eboot.bin")).unwrap(), b"EBOOT v2");
    let (ok, out) = cli(&["run", &s(&patch), "--data-dir", &s(&data)]);
    assert!(!ok && out.contains("not a game"), "{out}");

    // A package whose key is not derivable is refused as encrypted.
    let other = pkg(&dir, "retail.pkg", "01.000.000", b"x", false, "11111111111111111111111111111111");
    let (ok, out) = cli(&["install-pkg", &s(&other), "--data-dir", &s(&data)]);
    assert!(!ok && out.contains("encrypted, cannot be used"), "{out}");
    let (ok, out) = cli(&["info-pkg", &s(&other)]);
    assert!(ok && out.contains("encrypted, cannot be used"), "{out}");
    // ...unless its passcode is given.
    let (ok, out) = cli(&["info-pkg", &s(&other), "--passcode", "11111111111111111111111111111111"]);
    assert!(ok && out.contains("Status:      ready"), "{out}");

    let _ = std::fs::remove_dir_all(&dir);
}
