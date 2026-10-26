// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! PS4/PS5 package containers: fake PKGs, PFS images and PFSC streams.
//!
//! Supported inputs:
//! - **fake PKG** (`.pkg`, "\x7FCNT" container made by homebrew package
//!   tools): header, entry table (param.sfo, icon0.png, ...), the outer PFS
//!   (AES-XTS with a key derived from the content id and the package
//!   passcode, no console keys) and the inner `pfs_image.dat` (PFSC);
//! - **raw PFS image** (`.ffpkg`, `.pfs`, `.img`, `pfs_image.dat`), with or
//!   without a PFSC wrapper and with or without a nested `pfs_image.dat`.
//!
//! Retail-encrypted packages (PS4 packages whose key digest does not match
//! a fake-package key, PS5 "\x7FFIH" packages) are detected and refused
//! with [`Error::Encrypted`]. Nothing here contains or needs console keys.
//!
//! The launcher API is [`probe`] -> [`PackageInfo`]; the emulator opens a
//! [`Package`] and reads its [`Pfs`] tree directly or extracts it.

pub mod build;
pub mod crypto;
pub mod io;
pub mod param;
pub mod pfs;
pub mod pfsc;
pub mod pkg;

pub use io::{ReadAt, Source};
pub use param::TitleParams;
pub use pfs::{FileData, Node, NodeId, Pfs};
pub use pkg::PkgHeader;

use crate::io::{read_at, HostFile, Slice};
use crate::pfs::{is_pfs, PfsKey};
use crate::pfsc::{is_pfsc, Pfsc};
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Errors of this crate.
#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    /// Malformed or unrecognized data.
    Format(String),
    /// Encrypted content that cannot be read without console keys.
    Encrypted(String),
    /// A valid input this crate does not handle (yet).
    Unsupported(String),
    /// A path that does not exist inside the image.
    NotFound(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "I/O error: {e}"),
            Error::Format(m) => write!(f, "invalid package: {m}"),
            Error::Encrypted(m) => write!(f, "encrypted, cannot be used: {m}"),
            Error::Unsupported(m) => write!(f, "unsupported: {m}"),
            Error::NotFound(p) => write!(f, "{p}: no such file in the package"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        if e.kind() == std::io::ErrorKind::InvalidData {
            Error::Format(e.to_string())
        } else {
            Error::Io(e)
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// Outer format of a package file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Container {
    /// "\x7FCNT" package (fake or retail).
    Pkg,
    /// Bare PFS image.
    PfsImage,
    /// PFS image inside a PFSC stream.
    PfscImage,
}

/// What a package installs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContentKind {
    Game,
    /// A full (non-delta) update.
    Patch,
    /// A delta update (needs the base files to be patched in place).
    DeltaPatch,
    /// DLC / additional content.
    AdditionalContent,
    Unknown,
}

impl fmt::Display for ContentKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            ContentKind::Game => "game",
            ContentKind::Patch => "update",
            ContentKind::DeltaPatch => "delta update",
            ContentKind::AdditionalContent => "additional content (DLC)",
            ContentKind::Unknown => "unknown",
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Platform {
    Ps4,
    Ps5,
    Unknown,
}

/// Whether a package can be used.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Status {
    Ready,
    Encrypted(String),
    Unsupported(String),
}

/// Everything a launcher shows about a package.
#[derive(Clone, Debug)]
pub struct PackageInfo {
    pub path: PathBuf,
    pub container: Container,
    pub kind: ContentKind,
    pub platform: Platform,
    pub title_id: Option<String>,
    pub name: Option<String>,
    pub version: Option<String>,
    pub content_id: Option<String>,
    pub category: Option<String>,
    /// icon0.png bytes, when present.
    pub icon: Option<Vec<u8>>,
    /// Size of the package file.
    pub file_size: u64,
    pub status: Status,
}

impl PackageInfo {
    /// Title id usable as a folder name (sanitized), if known.
    pub fn folder_name(&self) -> Option<String> {
        let t = self.title_id.as_deref()?.trim();
        let ok = !t.is_empty() && t.len() <= 32 && t.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
        ok.then(|| t.to_ascii_uppercase())
    }
}

/// Options for opening packages.
#[derive(Clone, Debug)]
pub struct OpenOptions {
    /// Passcode of a fake package (32 characters).
    pub passcode: String,
}

impl Default for OpenOptions {
    fn default() -> Self {
        Self { passcode: crypto::DEFAULT_PASSCODE.to_string() }
    }
}

/// File extensions the launcher should consider.
pub const PACKAGE_EXTENSIONS: &[&str] = &["pkg", "ffpkg", "pfs"];

/// True when `path` has a package extension (case-insensitive).
pub fn has_package_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| PACKAGE_EXTENSIONS.iter().any(|x| x.eq_ignore_ascii_case(e)))
}

/// Recognize a package by its first bytes.
pub fn sniff(head: &[u8]) -> Option<Container> {
    if head.len() >= 4 && (head[..4] == pkg::PKG_MAGIC || head[..4] == pkg::PS5_RETAIL_MAGIC) {
        Some(Container::Pkg)
    } else if is_pfs(head) {
        Some(Container::PfsImage)
    } else if is_pfsc(head) {
        Some(Container::PfscImage)
    } else {
        None
    }
}

/// True when the file at `path` starts like a package or PFS image.
pub fn is_package_file(path: &Path) -> bool {
    let Ok(f) = fs::File::open(path) else { return false };
    let mut head = [0u8; 16];
    io::file_read_exact_at(&f, &mut head, 0).is_ok() && sniff(&head).is_some()
}

/// A recognized image format this crate does not read.
fn foreign_image(src: &dyn ReadAt) -> Option<&'static str> {
    let mut head = [0u8; 16];
    if read_at(src, &mut head, 0).ok()? >= 11 && &head[3..11] == b"EXFAT   " {
        return Some("exFAT disk image");
    }
    // UFS2: superblock at 64 KiB, magic at +0x55C.
    let mut m = [0u8; 4];
    if read_at(src, &mut m, 0x1_0000 + 0x55C).ok()? == 4 && u32::from_le_bytes(m) == 0x1954_0119 {
        return Some("UFS2 disk image");
    }
    None
}

/// An opened package with its game file tree.
pub struct Package {
    pub info: PackageInfo,
    /// Container header (PKG files only).
    pub header: Option<PkgHeader>,
    fs: Option<Arc<Pfs>>,
}

/// Metadata gathered while opening.
struct Opened {
    container: Container,
    header: Option<PkgHeader>,
    params: TitleParams,
    icon: Option<Vec<u8>>,
    fs: std::result::Result<Arc<Pfs>, Error>,
}

fn open_source(path: &Path) -> Result<(Source, u64)> {
    let f = fs::File::open(path)?;
    let src = HostFile::new(f)?;
    let len = src.len();
    Ok((Arc::new(src), len))
}

/// Descend into a nested `pfs_image.dat` (the outer PFS of a package holds
/// only that file; the game tree lives in the inner image).
fn descend(mut pfs: Pfs) -> Result<Pfs> {
    for _ in 0..2 {
        let root = pfs.root();
        let Some(id) = pfs.child(root, "pfs_image.dat") else { break };
        if pfs.child(root, "eboot.bin").is_some() || pfs.child(root, "sce_sys").is_some() {
            break;
        }
        let data = pfs.open_file(id)?.source();
        let mut head = [0u8; 16];
        read_at(&*data, &mut head, 0)?;
        let inner: Source = if is_pfsc(&head) { Arc::new(Pfsc::new(data)?) } else { data };
        pfs = Pfs::open(inner, None)?;
    }
    Ok(pfs)
}

fn open_image(src: Source, key: Option<PfsKey>) -> Result<Pfs> {
    descend(Pfs::open(src, key)?)
}

fn open_any(path: &Path, opts: &OpenOptions) -> Result<Opened> {
    let (src, len) = open_source(path)?;
    let mut head = [0u8; 16];
    read_at(&*src, &mut head, 0)?;
    match sniff(&head) {
        Some(Container::Pkg) => open_pkg(src, opts),
        Some(Container::PfsImage) => Ok(Opened {
            container: Container::PfsImage,
            header: None,
            params: TitleParams::default(),
            icon: None,
            fs: open_image(src, None).map(Arc::new),
        }),
        Some(Container::PfscImage) => {
            let fs = Pfsc::new(src).map_err(Error::from).and_then(|p| open_image(Arc::new(p), None)).map(Arc::new);
            Ok(Opened { container: Container::PfscImage, header: None, params: TitleParams::default(), icon: None, fs })
        }
        None => match foreign_image(&*src) {
            Some(kind) => Err(Error::Unsupported(format!("{kind} (only PKG and PFS images are supported)"))),
            None if len == 0 => Err(Error::Format("empty file".into())),
            None => Err(Error::Format("not a PKG or PFS image".into())),
        },
    }
}

fn open_pkg(src: Source, opts: &OpenOptions) -> Result<Opened> {
    let hdr = PkgHeader::parse(&*src)?;
    let mut params = hdr
        .read_entry(&*src, pkg::ENTRY_PARAM_SFO)
        .and_then(|b| param::parse_sfo(&b))
        .map(|m| TitleParams::from_sfo(&m))
        .unwrap_or_default();
    if params.content_id.is_none() && !hdr.content_id.is_empty() {
        params.content_id = Some(hdr.content_id.clone());
    }
    let icon = hdr.read_entry(&*src, pkg::ENTRY_ICON0_PNG);
    let fs = open_pkg_fs(&hdr, &src, opts).map(Arc::new);
    Ok(Opened { container: Container::Pkg, header: Some(hdr), params, icon, fs })
}

fn open_pkg_fs(hdr: &PkgHeader, src: &Source, opts: &OpenOptions) -> Result<Pfs> {
    if hdr.pfs_image_offset == 0 || hdr.pfs_image_size == 0 {
        return Err(Error::Unsupported("the package has no PFS image".into()));
    }
    let outer: Source = Arc::new(Slice::new(src.clone(), hdr.pfs_image_offset, hdr.pfs_image_size)?);
    let sb = pfs::Superblock::parse(&io::read_vec(&*outer, 0, pfs::SUPERBLOCK_SIZE)?)?;
    let key = if sb.is_encrypted() {
        let ekpfs = crypto::derive_key(&hdr.content_id, &opts.passcode, 1);
        if let Some(stored) = hdr.key_digest(&**src, 1) {
            if stored != crypto::key_digest(&ekpfs) {
                return Err(Error::Encrypted(
                    "retail package (or a fake package with a different passcode): its PFS key is not derivable".into(),
                ));
            }
        }
        Some(PfsKey { ekpfs, new_scheme: hdr.pfs_flags & pkg::PFS_FLAG_NEW_CRYPT != 0 })
    } else {
        None
    };
    open_image(outer, key)
}

fn classify(header: Option<&PkgHeader>, params: &TitleParams) -> ContentKind {
    let cat = params.category.as_deref().unwrap_or("").to_ascii_lowercase();
    if let Some(h) = header {
        if h.is_additional_content() || cat.starts_with("ac") {
            return ContentKind::AdditionalContent;
        }
        if h.is_delta_patch() {
            return ContentKind::DeltaPatch;
        }
        if h.is_patch() || cat.starts_with("gp") {
            return ContentKind::Patch;
        }
        if h.content_type == pkg::CONTENT_TYPE_GD || cat.starts_with("gd") {
            return ContentKind::Game;
        }
        return ContentKind::Unknown;
    }
    match cat.as_str() {
        c if c.starts_with("gp") => ContentKind::Patch,
        c if c.starts_with("ac") => ContentKind::AdditionalContent,
        _ => ContentKind::Game,
    }
}

fn read_small(pfs: &Pfs, path: &str, limit: u64) -> Option<Vec<u8>> {
    let id = pfs.lookup(path)?;
    let f = pfs.open_file(id).ok()?;
    (f.len() <= limit).then(|| f.read_all().ok()).flatten()
}

fn finish(path: &Path, file_size: u64, o: Opened) -> Package {
    let mut params = o.params;
    let mut icon = o.icon;
    let mut platform = Platform::Unknown;
    if let Ok(fs) = &o.fs {
        if let Some(p) = read_small(fs, "sce_sys/param.json", 1 << 20)
            .and_then(|b| TitleParams::from_json(&String::from_utf8_lossy(&b)))
        {
            platform = Platform::Ps5;
            let mut p = p;
            p.merge(&params);
            params = p;
        }
        if let Some(m) = read_small(fs, "sce_sys/param.sfo", 1 << 20).and_then(|b| param::parse_sfo(&b)) {
            params.merge(&TitleParams::from_sfo(&m));
            if platform == Platform::Unknown {
                platform = Platform::Ps4;
            }
        }
        if icon.is_none() {
            icon = read_small(fs, "sce_sys/icon0.png", 16 << 20);
        }
    }
    if platform == Platform::Unknown {
        platform = match params.title_id.as_deref().map(|t| t.get(..4).unwrap_or("").to_ascii_uppercase()) {
            Some(p) if p == "PPSA" || p == "ECAS" => Platform::Ps5,
            Some(p) if p == "CUSA" || p == "PCJS" || p == "PLJS" || p == "PLAS" => Platform::Ps4,
            _ if o.header.is_some() && params.title_id.is_some() => Platform::Ps4,
            _ => Platform::Unknown,
        };
    }
    let kind = classify(o.header.as_ref(), &params);
    let (status, fs) = match o.fs {
        Ok(fs) => (Status::Ready, Some(fs)),
        Err(Error::Encrypted(m)) => (Status::Encrypted(m), None),
        Err(e) => (Status::Unsupported(e.to_string()), None),
    };
    let info = PackageInfo {
        path: path.to_path_buf(),
        container: o.container,
        kind,
        platform,
        title_id: params.title_id,
        name: params.name,
        version: params.version,
        content_id: params.content_id,
        category: params.category,
        icon,
        file_size,
        status,
    };
    Package { info, header: o.header, fs }
}

/// Read a package's metadata for display. Encrypted or unreadable images
/// still return what the container exposes, with [`PackageInfo::status`]
/// saying why it cannot be used; only unrecognized files are errors.
pub fn probe(path: &Path) -> Result<PackageInfo> {
    probe_with(path, &OpenOptions::default())
}

pub fn probe_with(path: &Path, opts: &OpenOptions) -> Result<PackageInfo> {
    let size = fs::metadata(path)?.len();
    match open_any(path, opts) {
        Ok(o) => Ok(finish(path, size, o).info),
        Err(Error::Encrypted(m)) => Ok(PackageInfo {
            path: path.to_path_buf(),
            container: Container::Pkg,
            kind: ContentKind::Unknown,
            platform: Platform::Ps5,
            title_id: None,
            name: None,
            version: None,
            content_id: None,
            category: None,
            icon: None,
            file_size: size,
            status: Status::Encrypted(m),
        }),
        Err(e) => Err(e),
    }
}

/// One file or directory of a package.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileEntry {
    pub path: String,
    pub is_dir: bool,
    pub size: u64,
}

/// Extraction progress.
#[derive(Clone, Debug)]
pub struct Progress<'a> {
    pub path: &'a str,
    pub done_bytes: u64,
    pub total_bytes: u64,
    pub done_files: usize,
    pub total_files: usize,
}

impl Package {
    /// Open a package whose game files can be read. Encrypted packages and
    /// images without a readable file system are errors.
    pub fn open(path: &Path) -> Result<Self> {
        Self::open_with(path, &OpenOptions::default())
    }

    pub fn open_with(path: &Path, opts: &OpenOptions) -> Result<Self> {
        let size = fs::metadata(path)?.len();
        let o = open_any(path, opts)?;
        let fs_err = match &o.fs {
            Err(Error::Encrypted(m)) => return Err(Error::Encrypted(m.clone())),
            Err(e) => Some(e.to_string()),
            Ok(_) => None,
        };
        let p = finish(path, size, o);
        if p.fs.is_none() {
            return Err(Error::Unsupported(fs_err.unwrap_or_else(|| "no readable file system".into())));
        }
        Ok(p)
    }

    /// The game file tree (what `/app0` shows).
    pub fn fs(&self) -> Arc<Pfs> {
        self.fs.clone().expect("Package::open guarantees a file system")
    }

    /// Every file and directory, parents first.
    pub fn files(&self) -> Vec<FileEntry> {
        let fs = self.fs();
        fs.walk()
            .into_iter()
            .map(|id| {
                let n = fs.node(id);
                FileEntry { path: fs.path(id), is_dir: n.is_dir, size: n.size }
            })
            .collect()
    }

    /// Whole contents of one file.
    pub fn read(&self, path: &str) -> Result<Vec<u8>> {
        self.fs().read(path)
    }

    /// Write the game tree into `dest` (created if needed). Existing files
    /// are overwritten, so extracting an update over a game applies it.
    /// `sce_sys/param.sfo` and `sce_sys/icon0.png` are taken from the
    /// container entries when the image does not carry them.
    pub fn extract_to(&self, dest: &Path, progress: &mut dyn FnMut(&Progress)) -> Result<(usize, u64)> {
        use std::io::Write;
        let fs = self.fs();
        let ids = fs.walk();
        let total_bytes: u64 = ids.iter().map(|&i| fs.node(i).size).sum();
        let total_files = ids.iter().filter(|&&i| !fs.node(i).is_dir).count();
        fs::create_dir_all(dest)?;
        let (mut done_bytes, mut done_files) = (0u64, 0usize);
        let mut buf = vec![0u8; 4 << 20];
        for id in ids {
            let n = fs.node(id);
            let rel = fs.path(id);
            let out = dest.join(&rel);
            if n.is_dir {
                fs::create_dir_all(&out)?;
                continue;
            }
            progress(&Progress { path: &rel, done_bytes, total_bytes, done_files, total_files });
            let data = fs.open_file(id)?;
            let mut f = fs::File::create(&out)?;
            let mut off = 0u64;
            while off < data.len() {
                let got = data.read_at(&mut buf, off)?;
                if got == 0 {
                    break;
                }
                f.write_all(&buf[..got])?;
                off += got as u64;
                done_bytes += got as u64;
            }
            done_files += 1;
        }
        if let Some(h) = &self.header {
            let (src, _) = open_source(&self.info.path)?;
            let sys = dest.join("sce_sys");
            for (id, name) in [(pkg::ENTRY_PARAM_SFO, "param.sfo"), (pkg::ENTRY_ICON0_PNG, "icon0.png")] {
                if fs.lookup(&format!("sce_sys/{name}")).is_some() {
                    continue;
                }
                if let Some(bytes) = h.read_entry(&*src, id) {
                    fs::create_dir_all(&sys)?;
                    fs::write(sys.join(name), bytes)?;
                }
            }
        }
        progress(&Progress { path: "", done_bytes, total_bytes, done_files, total_files });
        Ok((done_files, done_bytes))
    }
}

#[cfg(test)]
mod tests;
