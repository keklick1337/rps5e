// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Tests of the disk-image readers (exFAT, UFS2, PFS containers) over
//! images built in memory, plus interoperability checks with the host's
//! exfatprogs when they are installed.

use super::*;
use crate::build::{ExFatBuilder, ExFatStore, PfsBuilder};
use crate::exfat::{DirEntry, ExFat};
use crate::image::{find_game_root, DiskKind};
use crate::io::{read_vec, Memory};
use crate::ufs::Ufs;

const PARAM_JSON: &str = r#"{"titleId":"PPSA99999","contentId":"UP9000-PPSA99999_00-TESTGAME00000000",
 "contentVersion":"01.020.000","localizedParameters":{"defaultLanguage":"en-US","en-US":{"titleName":"Image Game"}}}"#;
const LONG_NAME: &str = "a very long file name that needs several exFAT name entries to be stored.bin";

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

/// Scratch folder under the build directory (never the system /tmp).
fn scratch(name: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/ps5pkg-tests").join(format!("{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

fn mem(v: Vec<u8>) -> Source {
    Arc::new(Memory(v))
}

/// A game in a subfolder, with every storage mode and a decoy folder.
fn exfat_game() -> ExFatBuilder {
    let mut b = ExFatBuilder::new()
        .dir("Empty Folder")
        .file("readme.txt", b"not the game")
        .file("decoy/sce_sys/param.json", b"{}")
        .file("dumps/PPSA99999-app/eboot.bin", b"\x7FELF image eboot")
        .file("dumps/PPSA99999-app/sce_sys/param.json", PARAM_JSON.as_bytes())
        .file("dumps/PPSA99999-app/sce_sys/icon0.png", b"\x89PNG image icon")
        .file_source("dumps/PPSA99999-app/Data/Frag1.bin", mem(noise(0x5123, 1)), ExFatStore::Fragmented)
        .file_source("dumps/PPSA99999-app/Data/Frag2.bin", mem(noise(0x3001, 2)), ExFatStore::Fragmented)
        .file_source("dumps/PPSA99999-app/Data/Chain.bin", mem(noise(0x2800, 3)), ExFatStore::Chain)
        .file_source("dumps/PPSA99999-app/Data/Contig.bin", mem(noise(0x4000, 4)), ExFatStore::Contiguous)
        .file("dumps/PPSA99999-app/Data/empty.dat", b"")
        .file(&format!("dumps/PPSA99999-app/Data/{LONG_NAME}"), b"long")
        .file("dumps/PPSA99999-app/Data/Données/Été.txt", b"accents")
        .file("dumps/PPSA99999-app/a/b/c/d/e/deep.txt", b"deep");
    b.cluster_size = 0x1000;
    b
}

fn check_game(pfs: &Pfs) {
    assert_eq!(pfs.read("eboot.bin").unwrap(), b"\x7FELF image eboot");
    assert_eq!(pfs.read("SCE_SYS/Param.Json").unwrap(), PARAM_JSON.as_bytes(), "case-insensitive lookup");
    assert_eq!(pfs.read("Data/Frag1.bin").unwrap(), noise(0x5123, 1));
    assert_eq!(pfs.read("Data/Frag2.bin").unwrap(), noise(0x3001, 2));
    assert_eq!(pfs.read("Data/Chain.bin").unwrap(), noise(0x2800, 3));
    assert_eq!(pfs.read("Data/Contig.bin").unwrap(), noise(0x4000, 4));
    assert_eq!(pfs.read("Data/empty.dat").unwrap(), b"");
    assert_eq!(pfs.read(&format!("Data/{LONG_NAME}")).unwrap(), b"long");
    assert_eq!(pfs.read("Data/Données/Été.txt").unwrap(), b"accents");
    assert_eq!(pfs.read("a/b/c/d/e/deep.txt").unwrap(), b"deep");
    assert!(pfs.lookup("readme.txt").is_none(), "the tree is rooted at the game folder");
    // Reads across cluster boundaries of a fragmented chain.
    let f = pfs.open_path("Data/Frag1.bin").unwrap();
    let mut buf = vec![0u8; 0x2100];
    assert_eq!(f.read_at(&mut buf, 0xF80).unwrap(), 0x2100);
    assert_eq!(buf, noise(0x5123, 1)[0xF80..0x3080]);
    assert_eq!(f.read_at(&mut buf, 0x5000).unwrap(), 0x123);
    let d = pfs.lookup("Data").unwrap();
    assert!(pfs.node(d).is_dir);
    assert_eq!(pfs.node(pfs.lookup("Data/Contig.bin").unwrap()).size, 0x4000);
}

#[test]
fn exfat_volume_reads_every_storage_mode() {
    let img = exfat_game().build();
    let src = mem(img.to_vec());
    let fs = ExFat::open(src.clone()).unwrap();
    assert_eq!(fs.cluster_size(), 0x1000);
    // Native lookups compare names through the up-case table.
    let e = fs.lookup("DUMPS/ppsa99999-APP/data/frag2.BIN").unwrap().unwrap();
    assert_eq!(e.size, 0x3001);
    assert!(!e.contiguous);
    let c = fs.lookup("dumps/PPSA99999-app/Data/Contig.bin").unwrap().unwrap();
    assert!(c.contiguous);
    assert!(fs.lookup("dumps/missing").unwrap().is_none());
    assert!(fs.lookup("readme.txt/x").unwrap().is_none());

    let tree = fs.into_tree().unwrap();
    assert_eq!(tree.kind(), FsKind::ExFat);
    assert!(tree.superblock().is_none());
    let root = find_game_root(&tree).unwrap();
    assert_eq!(tree.path(root), "dumps/PPSA99999-app");
    let e = tree.lookup("Empty Folder").unwrap();
    assert!(tree.node(e).is_dir && tree.node(e).children.is_empty());
    check_game(&tree.subtree(root));

    // The disk-image entry point finds the same game folder.
    assert_eq!(image::detect(&src), Some(DiskKind::ExFat(0)));
    check_game(&image::open(src, DiskKind::ExFat(0)).unwrap());
}

#[test]
fn exfat_valid_data_length_reads_zeros() {
    let fs = ExFat::open(mem(exfat_game().build().to_vec())).unwrap();
    let mut e: DirEntry = fs.lookup("dumps/PPSA99999-app/Data/Chain.bin").unwrap().unwrap();
    e.valid = 0x1800;
    let data = fs.open_entry(&e).unwrap();
    assert_eq!(data.len(), 0x2800);
    let all = read_vec(&*data, 0, 0x2800).unwrap();
    assert_eq!(all[..0x1800], noise(0x2800, 3)[..0x1800]);
    assert!(all[0x1800..].iter().all(|&b| b == 0));
}

/// A file of `len` zero bytes ending in `tail`, generated on demand.
struct SparseTail {
    len: u64,
    tail: Vec<u8>,
}

impl ReadAt for SparseTail {
    fn len(&self) -> u64 {
        self.len
    }
    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
        let start = self.len - self.tail.len() as u64;
        for (i, b) in buf.iter_mut().enumerate() {
            let p = offset + i as u64;
            *b = if p >= start { self.tail[(p - start) as usize] } else { 0 };
        }
        Ok(())
    }
}

#[test]
fn exfat_file_larger_than_4_gib() {
    let len = (4u64 << 30) + 0x12345;
    for store in [ExFatStore::Contiguous, ExFatStore::Chain] {
        let big: Source = Arc::new(SparseTail { len, tail: b"END OF THE BIG FILE".to_vec() });
        let mut b = ExFatBuilder::new()
            .file("eboot.bin", b"e")
            .file("sce_sys/param.json", PARAM_JSON.as_bytes())
            .file_source("data/huge.pak", big, store);
        b.cluster_size = 1 << 20;
        let img: Source = Arc::new(b.build());
        assert!(img.len() > len);
        let pfs = image::open(img, DiskKind::ExFat(0)).unwrap();
        let f = pfs.open_path("data/huge.pak").unwrap();
        assert_eq!(f.len(), len);
        assert_eq!(pfs.node(pfs.lookup("data/huge.pak").unwrap()).size, len);
        let mut tail = [0u8; 19];
        assert_eq!(f.read_at(&mut tail, len - 19).unwrap(), 19);
        assert_eq!(&tail, b"END OF THE BIG FILE");
        let mut mid = [1u8; 8];
        f.read_at(&mut mid, 1 << 32).unwrap();
        assert_eq!(mid, [0; 8]);
    }
}

#[test]
fn exfat_checksums_are_enforced() {
    let good = exfat_game().build().to_vec();
    // A broken main boot region falls back to the backup one.
    let mut bad = good.clone();
    bad[0x64] ^= 0xFF;
    let fs = ExFat::open(mem(bad.clone())).unwrap();
    assert!(fs.lookup("dumps/PPSA99999-app/eboot.bin").unwrap().is_some());
    // Both broken: refused.
    bad[12 * 512 + 0x64] ^= 0xFF;
    assert!(matches!(ExFat::open(mem(bad)), Err(Error::Format(_))));
    // An entry set whose checksum does not match is skipped.
    let fs = ExFat::open(mem(good.clone())).unwrap();
    let root = fs.root_dir().unwrap();
    assert!(root.iter().any(|e| e.name == "readme.txt"));
    let mut bad = good.clone();
    let pos = bad
        .windows(20)
        .position(|w| w == "readme.txt".encode_utf16().flat_map(|u| u.to_le_bytes()).collect::<Vec<_>>().as_slice())
        .unwrap();
    bad[pos] ^= 1;
    let fs = ExFat::open(mem(bad)).unwrap();
    assert!(!fs.root_dir().unwrap().iter().any(|e| e.name.starts_with("readme") || e.name.starts_with("seadme")));
}

#[test]
fn exfat_corruption_never_panics() {
    let img = exfat_game().build().to_vec();
    for cut in [0x10, 0x200, 0x1800, 0x10000, img.len() / 2, img.len() - 1] {
        if let Ok(fs) = ExFat::open(mem(img[..cut].to_vec())) {
            let _ = fs.into_tree();
        }
    }
    for seed in 0..48u32 {
        let mut bad = img.clone();
        for (i, b) in noise(96, seed).iter().enumerate() {
            let at = 0x1_0000 + (seed as usize * 4099 + i * 977) % (bad.len() - 0x1_0000);
            bad[at] ^= b;
        }
        if let Ok(fs) = ExFat::open(mem(bad)) {
            if let Ok(t) = fs.into_tree() {
                for id in t.walk() {
                    if let Ok(f) = t.open_file(id) {
                        let _ = f.read_all();
                    }
                }
            }
        }
    }
}

#[test]
fn exfat_in_mbr_and_gpt_partitions() {
    let vol = exfat_game().build().to_vec();
    // MBR: one exFAT partition (type 0x07) at sector 2048.
    let mut mbr = vec![0u8; 2048 * 512];
    mbr[446 + 4] = 0x07;
    mbr[446 + 8..446 + 12].copy_from_slice(&2048u32.to_le_bytes());
    mbr[510] = 0x55;
    mbr[511] = 0xAA;
    mbr.extend_from_slice(&vol);
    let src = mem(mbr);
    assert_eq!(image::detect(&src), Some(DiskKind::ExFat(2048 * 512)));
    check_game(&image::open(src, DiskKind::ExFat(2048 * 512)).unwrap());
    // GPT: protective MBR, header at LBA 1, entries at LBA 2.
    let mut gpt = vec![0u8; 4096 * 512];
    gpt[446 + 4] = 0xEE;
    gpt[446 + 8..446 + 12].copy_from_slice(&1u32.to_le_bytes());
    gpt[510] = 0x55;
    gpt[511] = 0xAA;
    gpt[512..520].copy_from_slice(b"EFI PART");
    gpt[512 + 72..512 + 80].copy_from_slice(&2u64.to_le_bytes());
    gpt[512 + 80..512 + 84].copy_from_slice(&128u32.to_le_bytes());
    gpt[512 + 84..512 + 88].copy_from_slice(&128u32.to_le_bytes());
    let e = 1024 + 128; // second entry
    gpt[e + 32..e + 40].copy_from_slice(&4096u64.to_le_bytes());
    gpt.extend_from_slice(&vol);
    let src = mem(gpt);
    assert_eq!(image::detect(&src), Some(DiskKind::ExFat(4096 * 512)));
}

#[test]
fn exfat_package_files() {
    let dir = scratch("exfat-pkg");
    let path = dir.join("Image Game.exfat");
    exfat_game().build().write_to(&path).unwrap();
    assert!(has_package_extension(&path));
    assert!(is_package_file(&path));
    let info = probe(&path).unwrap();
    assert_eq!(info.container, Container::ExFatImage);
    assert_eq!(info.status, Status::Ready);
    assert_eq!(info.kind, ContentKind::Game);
    assert_eq!(info.platform, Platform::Ps5);
    assert_eq!(info.title_id.as_deref(), Some("PPSA99999"));
    assert_eq!(info.name.as_deref(), Some("Image Game"));
    assert_eq!(info.version.as_deref(), Some("01.020.000"));
    assert_eq!(info.icon.as_deref(), Some(&b"\x89PNG image icon"[..]));
    let p = Package::open(&path).unwrap();
    check_game(&p.fs());
    assert!(p.files().iter().any(|f| f.path == "Data/Frag2.bin" && f.size == 0x3001));
    let out = dir.join("extracted");
    p.extract_to(&out, &mut |_| {}).unwrap();
    assert_eq!(fs::read(out.join("Data/Frag1.bin")).unwrap(), noise(0x5123, 1));
    // A disk image without a game still opens, rooted at the volume.
    let path = dir.join("nogame.exfat");
    ExFatBuilder::new().file("x/y.txt", b"y").build().write_to(&path).unwrap();
    let p = Package::open(&path).unwrap();
    assert!(p.fs().lookup("x/y.txt").is_some());
    assert_eq!(p.info.title_id, None);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn ffpfsc_container_holds_an_exfat_image() {
    let dir = scratch("ffpfsc");
    let vol = exfat_game().build().to_vec();
    for compressed in [true, false] {
        let mut b = PfsBuilder::new();
        b = if compressed { b.compressed_file("PPSA99999 (01.020.000).exfat", &vol) } else { b.file("game.exfat", &vol) };
        b.version = pfs::PFS_VERSION_PS5;
        b.block_size = 0x10000;
        let path = dir.join(format!("game-{compressed}.ffpfsc"));
        fs::write(&path, b.build()).unwrap();
        assert!(is_package_file(&path));
        let p = Package::open(&path).unwrap();
        assert_eq!(p.info.container, Container::PfsContainer);
        assert_eq!(p.fs().kind(), FsKind::ExFat);
        assert_eq!(p.info.title_id.as_deref(), Some("PPSA99999"));
        check_game(&p.fs());
    }
    // The whole container PFSC-wrapped as well.
    let img = PfsBuilder::new().compressed_file("x.exfat", &vol).build();
    let path = dir.join("wrapped.ffpfsc");
    fs::write(&path, pfsc::build_pfsc(&img, 0x10000, true)).unwrap();
    check_game(&Package::open(&path).unwrap().fs());
    // A PFS image whose single file is not a disk image stays a PFS tree.
    let path = dir.join("plain.ffpfsc");
    fs::write(&path, PfsBuilder::new().file("notes.txt", b"hello").build()).unwrap();
    let p = Package::open(&path).unwrap();
    assert_eq!(p.info.container, Container::PfsImage);
    assert_eq!(p.read("notes.txt").unwrap(), b"hello");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn kraken_pfsc_is_reported_unsupported() {
    let mut s = pfsc::build_pfsc(&[7u8; 0x3000], 0x1000, false);
    s[4] = 3;
    let e = pfsc::Pfsc::new(mem(s)).err().unwrap();
    assert!(matches!(Error::from(e), Error::Unsupported(_)));
}

#[test]
fn game_root_search() {
    // Shallowest folder wins; param.sfo games are found when no param.json
    // game exists.
    let b = ExFatBuilder::new()
        .file("z/eboot.bin", b"1")
        .file("z/sce_sys/param.json", b"{}")
        .file("a/b/eboot.bin", b"2")
        .file("a/b/sce_sys/param.json", b"{}");
    let t = ExFat::open(mem(b.build().to_vec())).unwrap().into_tree().unwrap();
    assert_eq!(t.path(find_game_root(&t).unwrap()), "z");
    let b = ExFatBuilder::new().file("CUSA00001/eboot.bin", b"1").file("CUSA00001/sce_sys/param.sfo", b"sfo");
    let t = ExFat::open(mem(b.build().to_vec())).unwrap().into_tree().unwrap();
    assert_eq!(t.path(find_game_root(&t).unwrap()), "CUSA00001");
    let b = ExFatBuilder::new().file("eboot.bin", b"1").file("sce_sys/param.json", b"{}");
    let t = ExFat::open(mem(b.build().to_vec())).unwrap().into_tree().unwrap();
    assert_eq!(find_game_root(&t), Some(t.root()));
    let b = ExFatBuilder::new().file("x/eboot.bin", b"1");
    let t = ExFat::open(mem(b.build().to_vec())).unwrap().into_tree().unwrap();
    assert_eq!(find_game_root(&t), None);
}

/// True when an exfatprogs tool can be run.
fn have(tool: &str) -> bool {
    std::process::Command::new(tool).arg("-V").output().is_ok()
}

#[test]
fn exfat_reads_mkfs_exfat_volumes() {
    if !have("mkfs.exfat") {
        eprintln!("mkfs.exfat not installed, skipped");
        return;
    }
    let dir = scratch("mkfs");
    let path = dir.join("v.exfat");
    fs::File::create(&path).unwrap().set_len(32 << 20).unwrap();
    let ok = std::process::Command::new("mkfs.exfat").arg("-L").arg("RPS5E").arg(&path).output().unwrap().status.success();
    assert!(ok, "mkfs.exfat failed");
    let (src, _) = open_source(&path).unwrap();
    assert_eq!(image::detect(&src), Some(DiskKind::ExFat(0)));
    let fs = ExFat::open(src).unwrap();
    assert!(fs.root_dir().unwrap().is_empty());
    // The full up-case table of mkfs.exfat folds non-ASCII letters too.
    assert_eq!(fs.upcase_name("é"), fs.upcase_name("É"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn exfat_writer_passes_fsck_exfat() {
    if !have("fsck.exfat") {
        eprintln!("fsck.exfat not installed, skipped");
        return;
    }
    let dir = scratch("fsck");
    let path = dir.join("game.exfat");
    exfat_game().build().write_to(&path).unwrap();
    let out = std::process::Command::new("fsck.exfat").arg("-n").arg(&path).output().unwrap();
    assert!(
        out.status.success(),
        "fsck.exfat: {}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let _ = fs::remove_dir_all(&dir);
}

// ---- UFS2 ----

/// A minimal UFS2 writer: one cylinder group, inodes from fragment 72.
struct UfsImage {
    img: Vec<u8>,
    bsize: u64,
    fsize: u64,
    next_frag: u64,
    next_ino: u32,
}

const UFS_IBLKNO: u64 = 72;
const UFS_IPG: u64 = 64;

impl UfsImage {
    fn new(bsize: u64, fsize: u64, frags: u64) -> Self {
        let mut img = vec![0u8; (frags * fsize) as usize];
        let sb = ufs::SUPERBLOCK_OFFSET as usize;
        let put = |img: &mut Vec<u8>, o: usize, v: u32| img[sb + o..sb + o + 4].copy_from_slice(&v.to_le_bytes());
        put(&mut img, 0x08, 16);
        put(&mut img, 0x10, UFS_IBLKNO as u32);
        put(&mut img, 0x2C, 1);
        put(&mut img, 0x30, bsize as u32);
        put(&mut img, 0x34, fsize as u32);
        put(&mut img, 0x38, (bsize / fsize) as u32);
        put(&mut img, 0x74, (bsize / 8) as u32);
        put(&mut img, 0x78, (bsize / 256) as u32);
        put(&mut img, 0xB8, UFS_IPG as u32);
        put(&mut img, 0xBC, frags as u32);
        put(&mut img, ufs::MAGIC_OFFSET as usize, ufs::UFS2_MAGIC);
        let first_data = (UFS_IBLKNO * fsize + UFS_IPG * 256).div_ceil(bsize) * (bsize / fsize);
        UfsImage { img, bsize, fsize, next_frag: first_data, next_ino: 3 }
    }

    /// Store one block (or the fragments a short tail needs); returns its
    /// fragment number.
    fn block(&mut self, data: &[u8]) -> u64 {
        let f = self.next_frag;
        let o = (f * self.fsize) as usize;
        self.img[o..o + data.len()].copy_from_slice(data);
        self.next_frag += (data.len() as u64).div_ceil(self.fsize).max(1);
        // Keep later blocks block-aligned.
        let frag = self.bsize / self.fsize;
        self.next_frag = self.next_frag.next_multiple_of(frag);
        f
    }

    fn pointer_block(&mut self, ptrs: &[u64]) -> u64 {
        let mut b = vec![0u8; self.bsize as usize];
        for (i, p) in ptrs.iter().enumerate() {
            b[i * 8..i * 8 + 8].copy_from_slice(&p.to_le_bytes());
        }
        self.block(&b)
    }

    fn write_inode(&mut self, ino: u32, mode: u16, size: u64, direct: &[u64], indirect: [u64; 3]) {
        let o = (UFS_IBLKNO * self.fsize + ino as u64 * 256) as usize;
        let i = &mut self.img[o..o + 256];
        i[0..2].copy_from_slice(&mode.to_le_bytes());
        i[0x10..0x18].copy_from_slice(&size.to_le_bytes());
        i[0x28..0x30].copy_from_slice(&1_700_000_000i64.to_le_bytes());
        for (k, p) in direct.iter().enumerate() {
            i[0x70 + k * 8..0x78 + k * 8].copy_from_slice(&p.to_le_bytes());
        }
        for (k, p) in indirect.iter().enumerate() {
            i[0xD0 + k * 8..0xD8 + k * 8].copy_from_slice(&p.to_le_bytes());
        }
    }

    /// A regular file; `holes` are block indexes stored as holes. Blocks
    /// past the 12 direct ones go through a single-indirect block, then a
    /// double-indirect one.
    fn file(&mut self, data: &[u8], holes: &[usize]) -> u32 {
        let ino = self.next_ino;
        self.next_ino += 1;
        let bs = self.bsize as usize;
        let ptrs: Vec<u64> = data
            .chunks(bs)
            .enumerate()
            .map(|(i, c)| if holes.contains(&i) { 0 } else { self.block(c) })
            .collect();
        let per = bs / 8;
        let direct: Vec<u64> = ptrs.iter().take(12).copied().collect();
        let mut ind = [0u64; 3];
        if ptrs.len() > 12 {
            let single: Vec<u64> = ptrs[12..].iter().take(per).copied().collect();
            ind[0] = self.pointer_block(&single);
        }
        if ptrs.len() > 12 + per {
            let rest = &ptrs[12 + per..];
            let lvl1: Vec<u64> = rest.chunks(per).map(|c| c.to_vec()).collect::<Vec<_>>().iter().map(|c| self.pointer_block(c)).collect();
            ind[1] = self.pointer_block(&lvl1);
        }
        self.write_inode(ino, ufs::IFREG | 0o644, data.len() as u64, &direct, ind);
        ino
    }

    /// A directory with `entries` (name, inode, is_dir).
    fn dir(&mut self, ino: u32, parent: u32, entries: &[(&str, u32, bool)]) {
        let mut recs: Vec<(String, u32, u8)> = vec![(".".into(), ino, 4), ("..".into(), parent, 4)];
        recs.extend(entries.iter().map(|(n, i, d)| (n.to_string(), *i, if *d { 4 } else { 8 })));
        let mut data: Vec<u8> = Vec::new();
        let mut chunk: Vec<u8> = Vec::new();
        let mut last = 0usize;
        let flush = |data: &mut Vec<u8>, chunk: &mut Vec<u8>, last: usize| {
            let pad = 512 - chunk.len();
            let reclen = u16::from_le_bytes([chunk[last + 4], chunk[last + 5]]) as usize + pad;
            chunk[last + 4..last + 6].copy_from_slice(&(reclen as u16).to_le_bytes());
            chunk.resize(512, 0);
            data.append(chunk);
        };
        for (name, i, ty) in recs {
            let reclen = (8 + name.len() + 1).next_multiple_of(4);
            if chunk.len() + reclen > 512 {
                flush(&mut data, &mut chunk, last);
            }
            last = chunk.len();
            chunk.extend_from_slice(&i.to_le_bytes());
            chunk.extend_from_slice(&(reclen as u16).to_le_bytes());
            chunk.push(ty);
            chunk.push(name.len() as u8);
            chunk.extend_from_slice(name.as_bytes());
            chunk.resize(last + reclen, 0);
        }
        flush(&mut data, &mut chunk, last);
        let ptrs: Vec<u64> = data.chunks(self.bsize as usize).map(|c| self.block(c)).collect();
        self.write_inode(ino, ufs::IFDIR | 0o755, data.len() as u64, &ptrs, [0; 3]);
    }

    fn alloc_ino(&mut self) -> u32 {
        self.next_ino += 1;
        self.next_ino - 1
    }
}

/// A UFS2 game image: root = the game folder, a big file through the
/// double-indirect block with holes, a fragment tail and a symlink.
fn ufs_game() -> (Vec<u8>, Vec<u8>) {
    let mut u = UfsImage::new(4096, 1024, 4096);
    let big: Vec<u8> = {
        let mut v = noise(4096 * (12 + 512 + 7) + 777, 9);
        for hole in [3usize, 528] {
            v[hole * 4096..(hole + 1) * 4096].fill(0);
        }
        v
    };
    let eboot = u.file(b"\x7FELF ufs eboot", &[]);
    let param = u.file(PARAM_JSON.as_bytes(), &[]);
    let icon = u.file(b"\x89PNG ufs icon", &[]);
    let bigi = u.file(&big, &[3, 528]);
    let tail = u.file(&noise(4096 + 1500, 10), &[]);
    let link = u.alloc_ino();
    u.write_inode(link, 0o120777, 4, &[], [0; 3]);
    let sys = u.alloc_ino();
    let data = u.alloc_ino();
    let mut many: Vec<(String, u32, bool)> = Vec::new();
    for k in 0..40 {
        many.push((format!("entry number {k:02} with a longer name.dat"), u.file(format!("{k}").as_bytes(), &[]), false));
    }
    u.dir(sys, ufs::ROOT_INO, &[("param.json", param, false), ("icon0.png", icon, false)]);
    let mut d: Vec<(&str, u32, bool)> = vec![("Big.pak", bigi, false), ("Tail.bin", tail, false)];
    d.extend(many.iter().map(|(n, i, b)| (n.as_str(), *i, *b)));
    u.dir(data, ufs::ROOT_INO, &d);
    u.dir(
        ufs::ROOT_INO,
        ufs::ROOT_INO,
        &[("eboot.bin", eboot, false), ("sce_sys", sys, true), ("Data", data, true), ("link", link, false)],
    );
    (u.img, big)
}

#[test]
fn ufs2_tree_and_block_maps() {
    let (img, big) = ufs_game();
    let src = mem(img);
    assert_eq!(image::detect(&src), Some(DiskKind::Ufs2));
    let fs = Ufs::open(src.clone()).unwrap();
    assert_eq!(fs.block_size(), 4096);
    let t = image::open(src, DiskKind::Ufs2).unwrap();
    assert_eq!(t.kind(), FsKind::Ufs2);
    assert_eq!(t.read("eboot.bin").unwrap(), b"\x7FELF ufs eboot");
    assert_eq!(t.read("sce_sys/PARAM.json").unwrap(), PARAM_JSON.as_bytes());
    assert_eq!(t.read("Data/Big.pak").unwrap(), big);
    assert_eq!(t.read("Data/Tail.bin").unwrap(), noise(4096 + 1500, 10));
    assert_eq!(t.read("Data/entry number 39 with a longer name.dat").unwrap(), b"39");
    assert!(t.lookup("link").is_none(), "symbolic links are left out");
    assert_eq!(t.node(t.lookup("Data").unwrap()).children.len(), 42);
    let f = t.open_path("Data/Big.pak").unwrap();
    let mut buf = vec![0u8; 3 * 4096];
    f.read_at(&mut buf, 526 * 4096 + 100).unwrap();
    assert_eq!(buf, big[526 * 4096 + 100..529 * 4096 + 100]);
}

#[test]
fn ufs2_triple_indirect_sparse_file() {
    let mut u = UfsImage::new(4096, 4096, 512);
    let per = 512u64;
    let first_triple = 12 + per + per * per;
    let data = u.block(&[0xAB; 4096]);
    let l0 = u.pointer_block(&[data]);
    let l1 = u.pointer_block(&[l0]);
    let l2 = u.pointer_block(&[l1]);
    let size = (first_triple + 1) * 4096;
    let ino = 3;
    u.write_inode(ino, ufs::IFREG | 0o644, size, &[0; 12], [0, 0, l2]);
    u.dir(ufs::ROOT_INO, ufs::ROOT_INO, &[("sparse.bin", ino, false)]);
    let t = Ufs::open(mem(u.img)).unwrap().into_tree().unwrap();
    let f = t.open_path("sparse.bin").unwrap();
    assert_eq!(f.len(), size);
    let mut b = [0u8; 8];
    f.read_at(&mut b, first_triple * 4096 - 4).unwrap();
    assert_eq!(b, [0, 0, 0, 0, 0xAB, 0xAB, 0xAB, 0xAB]);
}

#[test]
fn ufs2_package_file() {
    let dir = scratch("ufs");
    let path = dir.join("game.ffpkg");
    fs::write(&path, ufs_game().0).unwrap();
    assert!(is_package_file(&path));
    let p = Package::open(&path).unwrap();
    assert_eq!(p.info.container, Container::Ufs2Image);
    assert_eq!(p.info.title_id.as_deref(), Some("PPSA99999"));
    assert_eq!(p.info.icon.as_deref(), Some(&b"\x89PNG ufs icon"[..]));
    assert_eq!(p.read("eboot.bin").unwrap(), b"\x7FELF ufs eboot");
    // Corruption never panics.
    let img = ufs_game().0;
    for seed in 0..32u32 {
        let mut bad = img.clone();
        for (i, b) in noise(64, seed).iter().enumerate() {
            let at = 0x1_0000 + (seed as usize * 7919 + i * 1031) % (bad.len() - 0x1_0000);
            bad[at] ^= b;
        }
        if let Ok(fs) = Ufs::open(mem(bad)) {
            if let Ok(t) = fs.into_tree() {
                for id in t.walk() {
                    if let Ok(f) = t.open_file(id) {
                        let _ = f.read_all();
                    }
                }
            }
        }
    }
    let _ = fs::remove_dir_all(&dir);
}

/// Build sample disk images of a game folder for manual runs:
/// `PS5PKG_IMAGE_SRC=<game folder> PS5PKG_IMAGE_OUT=<dir> cargo test
/// -p ps5pkg --release make_sample_images -- --ignored` writes
/// `<dir>/game.exfat` (the game in a subfolder) and `<dir>/game.ffpfsc`
/// (that image PFSC-compressed in a PS5 PFS container).
#[test]
#[ignore]
fn make_sample_images() {
    let (Ok(src), Ok(out)) = (std::env::var("PS5PKG_IMAGE_SRC"), std::env::var("PS5PKG_IMAGE_OUT")) else {
        eprintln!("set PS5PKG_IMAGE_SRC and PS5PKG_IMAGE_OUT");
        return;
    };
    let out = PathBuf::from(out);
    fs::create_dir_all(&out).unwrap();
    let name = Path::new(&src).file_name().unwrap().to_string_lossy().into_owned();
    let mut b = ExFatBuilder::new().host_dir(Path::new(&src), &format!("dump/{name}")).unwrap();
    b.cluster_size = 0x8000;
    let exfat = out.join("game.exfat");
    b.build().write_to(&exfat).unwrap();
    let vol = fs::read(&exfat).unwrap();
    let mut c = PfsBuilder::new().compressed_file(&format!("{name}.exfat"), &vol);
    c.version = pfs::PFS_VERSION_PS5;
    c.block_size = 0x10000;
    fs::write(out.join("game.ffpfsc"), c.build()).unwrap();
}
