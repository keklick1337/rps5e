// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Tests of the NAPS layout reader: layouts written by [`build_naps`] (stored
//! and zero spans, run bases with gaps in the packed data, continuation
//! ranges), corruption, and the real Silent Hill 2 package when present.

use super::*;
use crate::io::Memory;

fn pattern(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i.wrapping_mul(2654435761) >> 11) as u8).collect()
}

fn read_all(img: &NapsImage) -> Vec<u8> {
    let mut v = vec![0u8; img.len() as usize];
    img.read_exact_at(&mut v, 0).unwrap();
    v
}

#[test]
fn naps_stored_spans_with_run_bases_and_files() {
    let mut image = pattern(0x9_1234);
    // A zero region that is packed with the zero-block encoding.
    image[0x4_0000..0x8_0000].fill(0);
    let files = [0, 0x100, 0x2_3456, 0x4_0000, 0x8_0000];
    let (packed, layout) = build_naps(
        &image,
        &files,
        |i| if i == 3 { SpanForm::Zero } else { SpanForm::Stored },
        |i| if i == 2 || i == 5 { 0x123 } else { 0 },
    );
    let img = NapsImage::new(Arc::new(Memory(packed)), &layout).unwrap();
    assert_eq!(img.len(), image.len() as u64);
    assert_eq!(img.layout().file_starts(), &files);
    assert_eq!(read_all(&img), image);
    // Reads across span and file boundaries.
    for (off, n) in [(0xF0usize, 0x40usize), (0x2_3400, 0x1_0000), (0x3_FFF0, 0x5_0020), (0x9_1200, 0x34)] {
        let mut buf = vec![0u8; n];
        img.read_exact_at(&mut buf, off as u64).unwrap();
        assert_eq!(buf, image[off..off + n], "read at {off:#x}");
    }
    let mut buf = [0u8; 8];
    assert!(img.read_exact_at(&mut buf, image.len() as u64 - 4).is_err());
}

#[test]
fn naps_rejects_broken_layouts() {
    let image = pattern(0x5_0000);
    let (packed, layout) = build_naps(&image, &[0], |_| SpanForm::Stored, |_| 0);
    let data: Source = Arc::new(Memory(packed.clone()));
    assert!(NapsImage::new(data.clone(), &layout[..12]).is_err());
    // Truncated span table.
    assert!(NapsImage::new(data.clone(), &layout[..layout.len() - 24]).is_err());
    // pfs_image.dat shorter than the spans need.
    assert!(NapsImage::new(Arc::new(Memory(packed[..packed.len() - 1].to_vec())), &layout).is_err());
    // Random corruption never panics.
    let mut seed = 0x1234_5678_9ABC_DEFu64;
    for _ in 0..2000 {
        let mut l = layout.clone();
        for _ in 0..3 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let at = (seed >> 33) as usize % l.len();
            l[at] ^= 1 << ((seed >> 20) & 7);
        }
        if let Ok(img) = NapsImage::new(data.clone(), &l) {
            let mut buf = vec![0u8; 0x1000];
            let off = (seed >> 7) % img.len().max(1);
            let _ = crate::io::read_at(&img, &mut buf, off);
        }
    }
}

/// The Silent Hill 2 (PPSA08709) PS5 fake package when it is under
/// `GAMES_DIR`: the container opens, the tree comes from the Kraken inner
/// image, and files read back with their real headers.
#[test]
fn naps_real_package() {
    let Some(root) = std::env::var_os("GAMES_DIR") else { return };
    let Some(path) = crate::kraken::tests::find_file(std::path::Path::new(&root), crate::kraken::tests::REAL_PKG, 3) else {
        return;
    };
    let pkg = crate::Package::open(&path).expect("open the package");
    assert_eq!(pkg.info.title_id.as_deref(), Some("PPSA08709"));
    assert_eq!(pkg.info.kind, crate::ContentKind::Game);
    let fs = pkg.fs();
    let eboot = fs.open_path("eboot.bin").unwrap();
    let mut head = [0u8; 4];
    eboot.read_at(&mut head, 0).unwrap();
    assert_eq!(head, [0x54, 0x14, 0xF5, 0xEE], "eboot.bin is a SELF");
    let manifest = fs.read("package.manifest").unwrap();
    assert!(!manifest.is_empty());
    // The libc module decodes to an ELF-in-SELF as well.
    let libc = fs.read("sce_module/libc.prx").unwrap();
    assert_eq!(libc[..4], [0x54, 0x14, 0xF5, 0xEE]);
}

/// Local cross-check: SHA-256 and size of the files named (one inner path per
/// line) in `PS5PKG_HASH_LIST`, read from the package `PS5PKG_SH2_PKG`.
#[test]
#[ignore]
fn naps_hash_files() {
    use sha2::{Digest, Sha256};
    let (Some(pkg), Some(list)) = (std::env::var_os("PS5PKG_SH2_PKG"), std::env::var_os("PS5PKG_HASH_LIST")) else { return };
    let pkg = crate::Package::open(std::path::Path::new(&pkg)).unwrap();
    let fs = pkg.fs();
    for path in std::fs::read_to_string(list).unwrap().lines().filter(|l| !l.is_empty()) {
        let f = fs.open_path(path).unwrap();
        let mut h = Sha256::new();
        let mut buf = vec![0u8; 4 << 20];
        let mut off = 0u64;
        while off < f.len() {
            let n = f.read_at(&mut buf, off).unwrap();
            h.update(&buf[..n]);
            off += n as u64;
        }
        let hex: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
        println!("HASH {hex} {} {path}", f.len());
    }
}
