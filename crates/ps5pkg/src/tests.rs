// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! End-to-end tests over synthetic images built in memory.

use super::*;
use crate::build::{fake_pkg_image, PfsBuilder, PkgBuilder};
use crate::io::Memory;
use crate::param::{build_sfo, SfoValue};

const CID: &str = "UP9000-PPSA99999_00-TESTGAME00000000";
const PARAM_JSON: &str = r#"{"titleId":"PPSA99999","contentId":"UP9000-PPSA99999_00-TESTGAME00000000",
 "contentVersion":"01.000.000","localizedParameters":{"defaultLanguage":"en-US","en-US":{"titleName":"Test Game"}}}"#;

fn noise(len: usize, seed: u32) -> Vec<u8> {
    let mut x = seed.wrapping_mul(2654435761).wrapping_add(1);
    (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x as u8
        })
        .collect()
}

fn game_tree() -> PfsBuilder {
    PfsBuilder::new()
        .file("eboot.bin", b"\x7FELF fake eboot")
        .file("sce_sys/param.json", PARAM_JSON.as_bytes())
        .file("sce_sys/icon0.png", b"\x89PNG icon")
        .file("Data/Big.bin", &noise(0x4321, 1))
        .compressed_file("Data/Text.txt", &b"hello pfs ".repeat(2000))
        .dir("Empty")
}

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ps5pkg-test-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

fn check_tree(pfs: &Pfs) {
    assert_eq!(pfs.read("eboot.bin").unwrap(), b"\x7FELF fake eboot");
    assert_eq!(pfs.read("data/big.BIN").unwrap(), noise(0x4321, 1), "case-insensitive lookup");
    assert_eq!(pfs.read("Data/Text.txt").unwrap(), b"hello pfs ".repeat(2000));
    let text = pfs.lookup("Data/Text.txt").unwrap();
    assert_eq!(pfs.node(text).size, 20000, "compressed file reports its logical size");
    let e = pfs.lookup("Empty").unwrap();
    assert!(pfs.node(e).is_dir && pfs.node(e).children.is_empty());
    assert_eq!(pfs.lookup("Data/../sce_sys/./param.json"), pfs.lookup("sce_sys/param.json"));
    assert!(pfs.lookup("missing").is_none());
    assert!(pfs.lookup("eboot.bin/x").is_none());
    // Partial read across block boundaries.
    let f = pfs.open_path("Data/Big.bin").unwrap();
    let mut buf = vec![0u8; 0x1100];
    assert_eq!(f.read_at(&mut buf, 0xF80).unwrap(), 0x1100);
    assert_eq!(buf, noise(0x4321, 1)[0xF80..0x2080]);
    assert_eq!(f.read_at(&mut buf, 0x4320).unwrap(), 1);
    assert_eq!(f.read_at(&mut buf, 0x5000).unwrap(), 0);
}

#[test]
fn unsigned_pfs_image() {
    let img = game_tree().build();
    let pfs = Pfs::open(Arc::new(Memory(img)), None).unwrap();
    check_tree(&pfs);
    let paths: Vec<String> = pfs.walk().into_iter().map(|i| pfs.path(i)).collect();
    assert_eq!(
        paths,
        ["Data", "Data/Big.bin", "Data/Text.txt", "eboot.bin", "Empty", "sce_sys", "sce_sys/icon0.png", "sce_sys/param.json"]
    );
}

#[test]
fn signed_scattered_pfs_with_indirect_blocks() {
    let mut b = game_tree().file("Data/Huge.bin", &noise(20 * 0x1000 + 77, 9));
    b.scatter = true;
    let pfs = Pfs::open(Arc::new(Memory(b.build())), None).unwrap();
    check_tree(&pfs);
    assert_eq!(pfs.read("Data/Huge.bin").unwrap(), noise(20 * 0x1000 + 77, 9));
}

#[test]
fn encrypted_pfs_needs_the_right_key() {
    let ekpfs = crypto::derive_key(CID, crypto::DEFAULT_PASSCODE, 1);
    let mut b = game_tree();
    b.encrypt = Some((ekpfs, false, [7; 16]));
    let img = Arc::new(Memory(b.build()));
    assert!(matches!(Pfs::open(img.clone(), None), Err(Error::Encrypted(_))));
    let wrong = PfsKey { ekpfs: [1; 32], new_scheme: false };
    assert!(matches!(Pfs::open(img.clone(), Some(wrong)), Err(Error::Encrypted(_))));
    let pfs = Pfs::open(img, Some(PfsKey { ekpfs, new_scheme: false })).unwrap();
    check_tree(&pfs);
}

fn write_file(dir: &Path, name: &str, bytes: &[u8]) -> PathBuf {
    let p = dir.join(name);
    fs::write(&p, bytes).unwrap();
    p
}

fn sample_pkg(new_scheme: bool) -> PkgBuilder {
    let mut b = PkgBuilder::new(CID, fake_pkg_image(&game_tree(), CID, crypto::DEFAULT_PASSCODE, new_scheme));
    b.param_sfo = Some(build_sfo(&[
        ("APP_VER", SfoValue::Str("01.00".into())),
        ("CATEGORY", SfoValue::Str("gd".into())),
        ("TITLE", SfoValue::Str("Test Game SFO".into())),
        ("TITLE_ID", SfoValue::Str("PPSA99999".into())),
    ]));
    b.icon0 = Some(b"\x89PNG entry icon".to_vec());
    if new_scheme {
        b.pfs_flags = pkg::PFS_FLAG_NEW_CRYPT;
    }
    b
}

#[test]
fn fake_pkg_end_to_end() {
    let dir = tmp("pkg");
    for new_scheme in [false, true] {
        let path = write_file(&dir, "game.pkg", &sample_pkg(new_scheme).build());
        assert!(is_package_file(&path));
        let p = Package::open(&path).unwrap();
        assert_eq!(p.info.container, Container::Pkg);
        assert_eq!(p.info.kind, ContentKind::Game);
        assert_eq!(p.info.platform, Platform::Ps5);
        assert_eq!(p.info.title_id.as_deref(), Some("PPSA99999"));
        assert_eq!(p.info.name.as_deref(), Some("Test Game"), "param.json wins over param.sfo");
        assert_eq!(p.info.version.as_deref(), Some("01.000.000"));
        assert_eq!(p.info.content_id.as_deref(), Some(CID));
        assert_eq!(p.info.icon.as_deref(), Some(&b"\x89PNG entry icon"[..]));
        assert_eq!(p.info.status, Status::Ready);
        check_tree(&p.fs());
        let files = p.files();
        assert!(files.contains(&FileEntry { path: "Data/Big.bin".into(), is_dir: false, size: 0x4321 }));
    }

    // Extraction, plus the container's param.sfo dropped into sce_sys.
    let path = write_file(&dir, "game.pkg", &sample_pkg(false).build());
    let p = Package::open(&path).unwrap();
    let out = dir.join("out");
    let mut calls = 0;
    let (files, bytes) = p.extract_to(&out, &mut |_| calls += 1).unwrap();
    assert_eq!(files, 5);
    assert_eq!(bytes, 15 + PARAM_JSON.len() as u64 + 9 + 0x4321 + 20000);
    assert!(calls >= 5);
    assert_eq!(fs::read(out.join("Data/Big.bin")).unwrap(), noise(0x4321, 1));
    assert_eq!(fs::read(out.join("Data/Text.txt")).unwrap(), b"hello pfs ".repeat(2000));
    assert!(out.join("Empty").is_dir());
    assert!(out.join("sce_sys/param.sfo").is_file());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn retail_or_wrong_passcode_is_refused() {
    let dir = tmp("retail");
    // Key digests for another passcode: the PFS key is not derivable.
    let mut b = sample_pkg(false);
    b.passcode = Some("11111111111111111111111111111111".into());
    let path = write_file(&dir, "retail.pkg", &b.build());
    match Package::open(&path) {
        Err(Error::Encrypted(m)) => assert!(m.contains("retail")),
        other => panic!("expected Encrypted, got {:?}", other.map(|p| p.info)),
    }
    // probe still returns the container metadata for the launcher.
    let info = probe(&path).unwrap();
    assert!(matches!(info.status, Status::Encrypted(_)));
    assert_eq!(info.title_id.as_deref(), Some("PPSA99999"));
    assert_eq!(info.name.as_deref(), Some("Test Game SFO"));
    assert_eq!(info.icon.as_deref(), Some(&b"\x89PNG entry icon"[..]));

    // The right passcode opens it.
    let opts = OpenOptions { passcode: "11111111111111111111111111111111".into() };
    let img = fake_pkg_image(&game_tree(), CID, &opts.passcode, false);
    let mut b = PkgBuilder::new(CID, img);
    b.passcode = Some(opts.passcode.clone());
    let path = write_file(&dir, "custom.pkg", &b.build());
    assert!(Package::open(&path).is_err());
    assert!(Package::open_with(&path, &opts).is_ok());

    // No key digests at all: a wrong key is caught by the image check.
    let mut b = sample_pkg(false);
    b.passcode = None;
    b.pfs_image = fake_pkg_image(&game_tree(), CID, "22222222222222222222222222222222", false);
    let path = write_file(&dir, "nodigest.pkg", &b.build());
    assert!(matches!(Package::open(&path), Err(Error::Encrypted(_))));

    // PS5 retail container.
    let mut raw = vec![0u8; 0x2000];
    raw[..4].copy_from_slice(&pkg::PS5_RETAIL_MAGIC);
    let path = write_file(&dir, "ps5retail.pkg", &raw);
    assert!(matches!(Package::open(&path), Err(Error::Encrypted(_))));
    assert!(matches!(probe(&path).unwrap().status, Status::Encrypted(_)));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn update_and_dlc_are_classified() {
    let dir = tmp("kinds");
    let mut b = sample_pkg(false);
    b.content_flags = pkg::CONTENT_FLAG_FIRST_PATCH;
    let path = write_file(&dir, "patch.pkg", &b.build());
    assert_eq!(Package::open(&path).unwrap().info.kind, ContentKind::Patch);
    b.content_flags = pkg::CONTENT_FLAG_DELTA_PATCH;
    let path = write_file(&dir, "delta.pkg", &b.build());
    assert_eq!(Package::open(&path).unwrap().info.kind, ContentKind::DeltaPatch);
    b.content_flags = 0;
    b.content_type = pkg::CONTENT_TYPE_AC;
    let path = write_file(&dir, "dlc.pkg", &b.build());
    assert_eq!(Package::open(&path).unwrap().info.kind, ContentKind::AdditionalContent);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn raw_and_pfsc_images() {
    let dir = tmp("raw");
    let img = game_tree().build();
    let path = write_file(&dir, "game.ffpkg", &img);
    let p = Package::open(&path).unwrap();
    assert_eq!(p.info.container, Container::PfsImage);
    assert_eq!(p.info.title_id.as_deref(), Some("PPSA99999"));
    assert_eq!(p.info.icon.as_deref(), Some(&b"\x89PNG icon"[..]));
    check_tree(&p.fs());

    let path = write_file(&dir, "pfs_image.dat", &pfsc::build_pfsc(&img, 0x10000, true));
    let p = Package::open(&path).unwrap();
    assert_eq!(p.info.container, Container::PfscImage);
    check_tree(&p.fs());

    // A raw image that nests the game in pfs_image.dat (an outer image
    // without encryption).
    let mut outer = PfsBuilder::new().file("pfs_image.dat", &pfsc::build_pfsc(&img, 0x1000, false));
    outer.signed = true;
    let path = write_file(&dir, "nested.ffpkg", &outer.build());
    check_tree(&Package::open(&path).unwrap().fs());

    // Unrecognized and foreign formats.
    let path = write_file(&dir, "junk.pkg", b"definitely not a package");
    assert!(matches!(probe(&path), Err(Error::Format(_))));
    let mut exfat = vec![0u8; 512];
    exfat[3..11].copy_from_slice(b"EXFAT   ");
    let path = write_file(&dir, "game.exfat", &exfat);
    assert!(matches!(Package::open(&path), Err(Error::Unsupported(_))));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn corrupt_images_fail_cleanly() {
    let img = game_tree().build();
    // Truncations at many points must never panic.
    for cut in [0x10, 0x400, 0x1000, 0x1800, img.len() / 2, img.len() - 1] {
        let _ = Pfs::open(Arc::new(Memory(img[..cut].to_vec())), None);
    }
    // Random corruption of the inode/dirent area.
    for seed in 0..32u32 {
        let mut bad = img.clone();
        for (i, b) in noise(64, seed).iter().enumerate() {
            let at = 0x1000 + (seed as usize * 97 + i * 131) % (bad.len() - 0x1000);
            bad[at] ^= b;
        }
        if let Ok(p) = Pfs::open(Arc::new(Memory(bad)), None) {
            for id in p.walk() {
                if let Ok(f) = p.open_file(id) {
                    let _ = f.read_all();
                }
            }
        }
    }
}

/// A tree without sce_sys (a PS5 image keeps param.json in the container).
fn ps5_game_tree() -> PfsBuilder {
    PfsBuilder::new()
        .file("eboot.bin", b"\x7FELF fake eboot")
        .file("Data/Big.bin", &noise(0x4321, 1))
        .file("Data/Zeros.bin", &vec![0u8; 0x8_0000])
        .compressed_file("Data/Text.txt", &b"hello pfs ".repeat(30000))
        .dir("Empty")
}

#[test]
fn ps5_fake_package_opens_through_naps() {
    let dir = tmp("ps5fih");
    let path = dir.join("game.pkg");
    fs::write(&path, crate::build::ps5_fake_pkg(&ps5_game_tree(), CID, PARAM_JSON.as_bytes(), Some(b"\x89PNG icon"), false))
        .unwrap();
    let info = probe(&path).unwrap();
    assert_eq!(info.status, Status::Ready);
    assert_eq!(info.platform, Platform::Ps5);
    assert_eq!(info.kind, ContentKind::Game);
    assert_eq!(info.title_id.as_deref(), Some("PPSA99999"));
    assert_eq!(info.name.as_deref(), Some("Test Game"));
    assert_eq!(info.icon.as_deref(), Some(&b"\x89PNG icon"[..]));
    let p = Package::open(&path).unwrap();
    let fs = p.fs();
    assert_eq!(fs.read("eboot.bin").unwrap(), b"\x7FELF fake eboot");
    assert_eq!(fs.read("Data/Big.bin").unwrap(), noise(0x4321, 1));
    assert_eq!(fs.read("Data/Zeros.bin").unwrap(), vec![0u8; 0x8_0000]);
    // A Kraken-format PFSC file: logical size and contents.
    let text = fs.lookup("Data/Text.txt").unwrap();
    assert_eq!(fs.node(text).size, 300000);
    assert_eq!(fs.read("Data/Text.txt").unwrap(), b"hello pfs ".repeat(30000));
    // The container's plain entries appear under sce_sys like an installed title.
    assert_eq!(fs.read("sce_sys/param.json").unwrap(), PARAM_JSON.as_bytes());
    assert_eq!(fs.read("sce_sys/icon0.png").unwrap(), b"\x89PNG icon");
    assert!(fs.node(fs.lookup("Empty").unwrap()).is_dir);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn ps5_retail_and_encrypted_packages_are_refused() {
    let dir = tmp("ps5retail");
    // The retail byte.
    let path = dir.join("retail.pkg");
    fs::write(&path, crate::build::ps5_fake_pkg(&ps5_game_tree(), CID, PARAM_JSON.as_bytes(), None, true)).unwrap();
    assert!(matches!(Package::open(&path), Err(Error::Encrypted(_))));
    let info = probe(&path).unwrap();
    assert!(matches!(info.status, Status::Encrypted(_)));
    assert_eq!(info.title_id.as_deref(), Some("PPSA99999"), "metadata still shows");
    // An outer image with a real (non plain-text) seed.
    let mut raw = crate::build::ps5_fake_pkg(&ps5_game_tree(), CID, PARAM_JSON.as_bytes(), None, false);
    let sb = u64::from_le_bytes(raw[0x20..0x28].try_into().unwrap()) as usize;
    raw[sb + 0x370..sb + 0x380].copy_from_slice(&[0x5A; 16]);
    let path = dir.join("enc.pkg");
    fs::write(&path, &raw).unwrap();
    assert!(matches!(Package::open(&path), Err(Error::Encrypted(_))));
    let _ = fs::remove_dir_all(&dir);
}
