// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! `info-pkg` and `install-pkg`.

use anyhow::{anyhow, Result};
use ps5pkg::{Container, ContentKind, OpenOptions, Package, Platform, Status};
use std::io::Write;
use std::path::{Path, PathBuf};

fn container_name(c: Container) -> &'static str {
    match c {
        Container::Pkg => "PKG",
        Container::PfsImage => "PFS image",
        Container::PfscImage => "PFSC-compressed PFS image",
    }
}

fn human(n: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = n as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 { format!("{n} B") } else { format!("{v:.1} {}", UNITS[u]) }
}

/// Open a package for reading its files, turning the crate's refusals
/// into one clear message.
pub fn open_game(path: &Path, opts: &OpenOptions) -> Result<Package> {
    Package::open_with(path, opts).map_err(|e| match e {
        ps5pkg::Error::Encrypted(m) => anyhow!("{}: encrypted, cannot be used ({m})", path.display()),
        e => anyhow!("{}: {e}", path.display()),
    })
}

pub fn info(path: &str, opts: &OpenOptions, list_files: bool) -> Result<()> {
    let p = Path::new(path);
    let info = ps5pkg::probe_with(p, opts).map_err(|e| anyhow!("{path}: {e}"))?;
    let opt = |v: &Option<String>| v.clone().unwrap_or_else(|| "-".into());
    println!("File:        {} ({})", path, human(info.file_size));
    println!("Format:      {}", container_name(info.container));
    match &info.status {
        Status::Ready => println!("Status:      ready"),
        Status::Encrypted(m) => println!("Status:      encrypted, cannot be used ({m})"),
        Status::Unsupported(m) => println!("Status:      unsupported ({m})"),
    }
    println!("Content:     {}", info.kind);
    let platform = match info.platform {
        Platform::Ps5 => "PS5",
        Platform::Ps4 => "PS4",
        Platform::Unknown => "unknown",
    };
    println!("Platform:    {platform}");
    println!("Title ID:    {}", opt(&info.title_id));
    println!("Name:        {}", opt(&info.name));
    println!("Version:     {}", opt(&info.version));
    println!("Content ID:  {}", opt(&info.content_id));
    println!("Category:    {}", opt(&info.category));
    println!("Icon:        {}", info.icon.as_ref().map_or("-".into(), |i| human(i.len() as u64)));
    if info.status != Status::Ready {
        return Ok(());
    }
    let pkg = open_game(p, opts)?;
    if let Some(h) = &pkg.header {
        println!(
            "PKG:         content type {:#x}, flags {:#010x}, DRM {:#x}, PFS at {:#x} ({})",
            h.content_type,
            h.content_flags,
            h.drm_type,
            h.pfs_image_offset,
            human(h.pfs_image_size)
        );
        println!("Entries:");
        for e in &h.entries {
            println!(
                "  {:#06x} {:>10}  {}{}",
                e.id,
                e.size,
                e.name,
                if e.is_encrypted() { "  (encrypted)" } else { "" }
            );
        }
    }
    let fs = pkg.fs();
    let sb = fs.superblock();
    println!(
        "PFS:         block size {:#x}, {} inodes, {}{}",
        sb.block_size,
        sb.inode_count,
        if sb.mode & ps5pkg::pfs::MODE_SIGNED != 0 { "signed" } else { "unsigned" },
        if sb.mode & ps5pkg::pfs::MODE_64BIT != 0 { ", 64-bit" } else { "" }
    );
    let files = pkg.files();
    let count = files.iter().filter(|f| !f.is_dir).count();
    let total: u64 = files.iter().map(|f| f.size).sum();
    println!("Files:       {count} files, {}", human(total));
    if list_files {
        for f in &files {
            if f.is_dir {
                println!("  {:>14}  {}/", "", f.path);
            } else {
                println!("  {:>14}  {}", f.size, f.path);
            }
        }
    }
    Ok(())
}

/// Title id recorded in an installed game folder.
fn installed_title(dir: &Path) -> Option<String> {
    let json = std::fs::read_to_string(dir.join("sce_sys/param.json")).ok()?;
    ps5pkg::TitleParams::from_json(&json)?.title_id
}

pub fn install(path: &str, into: Option<&Path>, data: &Path, opts: &OpenOptions) -> Result<()> {
    let p = Path::new(path);
    let pkg = open_game(p, opts)?;
    let title = pkg.info.folder_name();
    match pkg.info.kind {
        ContentKind::Game | ContentKind::Unknown | ContentKind::Patch => {}
        ContentKind::DeltaPatch => {
            return Err(anyhow!("{path}: delta updates are not supported yet (use a full update package)"))
        }
        ContentKind::AdditionalContent => return Err(anyhow!("{path}: DLC packages are not supported yet")),
    }
    let dest: PathBuf = match into {
        Some(d) => d.to_path_buf(),
        None => {
            let t = title.clone().ok_or_else(|| anyhow!("{path}: no title id in the package; pass --into <dir>"))?;
            data.join("games").join(t)
        }
    };
    if pkg.info.kind == ContentKind::Patch {
        if !dest.join("eboot.bin").is_file() {
            return Err(anyhow!(
                "{path} is an update for {}; install the game first (expected it in {})",
                title.as_deref().unwrap_or("?"),
                dest.display()
            ));
        }
        if let (Some(base), Some(t)) = (installed_title(&dest), pkg.info.title_id.as_deref()) {
            if !base.eq_ignore_ascii_case(t) {
                return Err(anyhow!("{path} updates {t}, but {} holds {base}", dest.display()));
            }
        }
        println!("applying update {} to {}", pkg.info.version.as_deref().unwrap_or("?"), dest.display());
    } else if dest.join("eboot.bin").exists() {
        println!("{} already holds a game; files are overwritten", dest.display());
    }
    println!(
        "installing {} {} ({}) to {}",
        title.as_deref().unwrap_or("?"),
        pkg.info.name.as_deref().unwrap_or(""),
        pkg.info.version.as_deref().unwrap_or("?"),
        dest.display()
    );
    let mut last_pct = u64::MAX;
    let (files, bytes) = pkg
        .extract_to(&dest, &mut |pr| {
            let pct = if pr.total_bytes == 0 { 100 } else { pr.done_bytes * 100 / pr.total_bytes };
            if pct != last_pct {
                last_pct = pct;
                eprint!("\r  {pct:>3}%  {}/{} files", pr.done_files, pr.total_files);
                let _ = std::io::stderr().flush();
            }
        })
        .map_err(|e| anyhow!("extract {path}: {e}"))?;
    eprintln!();
    println!("installed {files} files ({}) into {}", human(bytes), dest.display());
    println!("run it with: rps5e-cli run \"{}\"", dest.display());
    Ok(())
}
