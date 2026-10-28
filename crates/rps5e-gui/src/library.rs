// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Game library: finds game folders, package files (`.pkg`, `.ffpkg`) and
//! disk images (`.exfat`, `.ffpfsc`)
//! under the library folders (recursively) and reads their metadata:
//! `sce_sys/param.json` (PS5) or `sce_sys/param.sfo`, `sce_sys/icon0.png`;
//! for a package what `ps5pkg::probe` reads from it (title, title id,
//! version, icon) and whether it can be used: a fake package runs as it is,
//! a retail-encrypted one cannot be run.
//!
//! A folder scan lists only launchable games: a folder with `eboot.bin` and
//! a param file whose category is a game application (not additional
//! content, an update or a system application), and runnable game
//! packages. Loose executables, DLC, patch and fix folders are not games.
//! One card per title id: the most complete copy is kept. Single entries
//! added by hand are listed as long as they can be read.

use std::path::{Path, PathBuf};

/// Directories deeper than this below a library folder are not searched.
pub const MAX_DEPTH: usize = 8;

/// Package and disk image file extensions.
pub const PACKAGE_EXTENSIONS: [&str; 4] = ["pkg", "ffpkg", "exfat", "ffpfsc"];

/// Where a library entry comes from.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum GameSource {
    /// An extracted game folder holding the executable (the game's /app0).
    Folder(PathBuf),
    /// A package file; it has to be installed before it can run.
    Package(PathBuf),
    /// A single executable (`.elf`, `.self`, `.bin`) added by hand.
    Executable(PathBuf),
}

impl Default for GameSource {
    fn default() -> Self {
        GameSource::Folder(PathBuf::new())
    }
}

/// Whether a package can be used, as `rps5e-cli info-pkg` reports it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PackageState {
    /// A fake game package: runs directly (mounted as /app0) or installs.
    Runnable,
    /// Readable, but not a game (an update or DLC): installed over a game.
    NotAGame(String),
    /// Retail-encrypted (or a fake package with another passcode).
    Encrypted(String),
    Unsupported(String),
    /// Not a package the emulator recognizes.
    Unreadable(String),
}

impl PackageState {
    pub fn can_run(&self) -> bool {
        matches!(self, PackageState::Runnable)
    }

    /// Short state for the library.
    pub fn label(&self) -> &'static str {
        match self {
            PackageState::Runnable => "package, runnable",
            PackageState::NotAGame(_) => "package, not a game",
            PackageState::Encrypted(_) => "encrypted, cannot be run",
            PackageState::Unsupported(_) => "package, unsupported",
            PackageState::Unreadable(_) => "package, unreadable",
        }
    }

    /// The reason behind the state.
    pub fn detail(&self) -> String {
        match self {
            PackageState::Runnable => "fake package: runs directly, or install it into a game folder".into(),
            PackageState::NotAGame(m) | PackageState::Encrypted(m) | PackageState::Unsupported(m) | PackageState::Unreadable(m) => m.clone(),
        }
    }
}

impl GameSource {
    pub fn path(&self) -> &Path {
        match self {
            GameSource::Folder(p) | GameSource::Package(p) | GameSource::Executable(p) => p,
        }
    }

    pub fn is_package(&self) -> bool {
        matches!(self, GameSource::Package(_))
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Game {
    pub source: GameSource,
    /// Folder of the game (the package's folder for a package).
    pub dir: PathBuf,
    /// `eboot.bin` (empty for a package).
    pub eboot: PathBuf,
    pub title: String,
    pub title_id: Option<String>,
    /// Content version.
    pub version: Option<String>,
    /// Application version (`appVersion`, else the content version), the
    /// version game patches name.
    pub app_version: Option<String>,
    /// Required system software, e.g. `11.60`.
    pub firmware: Option<String>,
    /// `icon0.png`; for a package a name of its own (`<package>#icon0.png`)
    /// whose bytes the scan returns.
    pub icon: Option<PathBuf>,
    /// A package's state (None for game folders).
    pub package: Option<PackageState>,
}

impl Game {
    /// Stable key of the entry (its path), used for per-game configurations.
    pub fn key(&self) -> String {
        self.source.path().display().to_string()
    }

    /// Whether Run can start it: a game folder, or a runnable package.
    pub fn can_run(&self) -> bool {
        match &self.source {
            GameSource::Folder(_) | GameSource::Executable(_) => true,
            GameSource::Package(_) => self.package.as_ref().is_some_and(PackageState::can_run),
        }
    }

    /// The executable a run starts (`elf` inside the game folder).
    pub fn executable(&self, elf: &str) -> PathBuf {
        if matches!(self.source, GameSource::Executable(_)) || elf.is_empty() || elf == crate::config::DEFAULT_ELF {
            self.eboot.clone()
        } else {
            self.dir.join(elf)
        }
    }
}

/// What a param file says the content is.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum ContentCategory {
    /// A game application: `applicationCategoryType` 0, `CATEGORY` `gd*`.
    Game,
    /// Additional content (DLC): a PS5 param.json without
    /// `applicationCategoryType` (applications must have it), `CATEGORY` `ac`.
    AdditionalContent,
    /// An update (`CATEGORY` `gp*`).
    Patch,
    /// Another application type (system or media application).
    Other(String),
    /// A param.sfo without `CATEGORY`.
    #[default]
    Unknown,
}

impl ContentCategory {
    /// Whether a folder of this category is a game of its own.
    pub fn is_game(&self) -> bool {
        matches!(self, ContentCategory::Game | ContentCategory::Unknown)
    }

    /// From `applicationCategoryType` of a PS5 param.json.
    pub fn from_json(v: Option<&serde_json::Value>) -> Self {
        match v {
            None => ContentCategory::AdditionalContent,
            Some(v) if v.as_u64() == Some(0) => ContentCategory::Game,
            Some(v) => ContentCategory::Other(format!("applicationCategoryType {v}")),
        }
    }

    /// From `CATEGORY` of a param.sfo.
    pub fn from_sfo(c: &str) -> Self {
        let c = c.trim().to_ascii_lowercase();
        if c.starts_with("gd") {
            ContentCategory::Game
        } else if c.starts_with("gp") {
            ContentCategory::Patch
        } else if c.starts_with("ac") {
            ContentCategory::AdditionalContent
        } else {
            ContentCategory::Other(format!("CATEGORY {c}"))
        }
    }

    /// What the content is, for messages.
    pub fn describe(&self) -> String {
        match self {
            ContentCategory::Game | ContentCategory::Unknown => "game".into(),
            ContentCategory::AdditionalContent => "additional content (DLC)".into(),
            ContentCategory::Patch => "update".into(),
            ContentCategory::Other(m) => format!("not a game application ({m})"),
        }
    }
}

/// Metadata fields read from param.json / param.sfo.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ParamInfo {
    pub title: Option<String>,
    pub title_id: Option<String>,
    pub version: Option<String>,
    pub app_version: Option<String>,
    pub firmware: Option<String>,
    pub category: ContentCategory,
}

/// `requiredSystemSoftwareVersion` (`0xMMmmpp...`) as `MM.mm[.pp]`.
pub fn firmware_version(encoded: &str) -> Option<String> {
    let hex = encoded.trim().strip_prefix("0x").or_else(|| encoded.trim().strip_prefix("0X"))?;
    if hex.len() != 16 || !hex.bytes().all(|c| c.is_ascii_hexdigit()) || !hex[..6].bytes().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let major: u32 = hex[..2].parse().ok()?;
    let mut v = format!("{major}.{}", &hex[2..4]);
    if &hex[4..6] != "00" {
        v.push('.');
        v.push_str(&hex[4..6]);
    }
    Some(v)
}

fn non_empty(s: Option<&str>) -> Option<String> {
    s.map(str::trim).filter(|s| !s.is_empty()).map(str::to_string)
}

/// Parse a PS5 `param.json`: `titleId`; version from `contentVersion`,
/// `masterVersion` or `targetContentVersion`; title from
/// `localizedParameters[defaultLanguage].titleName`, else `en-US`, else
/// the first language that has one.
pub fn parse_param_json(text: &str) -> Option<ParamInfo> {
    let v: serde_json::Value = serde_json::from_str(text.trim_start_matches('\u{FEFF}')).ok()?;
    let s = |k: &str| non_empty(v.get(k).and_then(|x| x.as_str()));
    let version = s("contentVersion").or_else(|| s("masterVersion")).or_else(|| s("targetContentVersion"));
    let loc = v.get("localizedParameters").or_else(|| v.get("disc").and_then(|d| d.get("localizedParameters")));
    let title = loc.and_then(|loc| {
        let name = |lang: &str| non_empty(loc.get(lang).and_then(|l| l.get("titleName")).and_then(|t| t.as_str()));
        let default = loc.get("defaultLanguage").and_then(|d| d.as_str()).and_then(name);
        default.or_else(|| name("en-US")).or_else(|| {
            loc.as_object()?
                .values()
                .find_map(|l| non_empty(l.get("titleName").and_then(|t| t.as_str())))
        })
    });
    let app_version = s("appVersion").or_else(|| s("contentVersion"));
    let firmware = s("requiredSystemSoftwareVersion").as_deref().and_then(firmware_version);
    let category = ContentCategory::from_json(v.get("applicationCategoryType"));
    Some(ParamInfo { title, title_id: s("titleId"), version, app_version, firmware, category })
}

/// Parse a binary `param.sfo` (PSF): header `\0PSF`, version, key table
/// offset, data table offset, entry count; 16-byte entries {key offset u16,
/// format u16, length u32, max length u32, data offset u32}. Reads TITLE,
/// TITLE_ID and APP_VER (else VERSION).
pub fn parse_param_sfo(b: &[u8]) -> Option<ParamInfo> {
    let u16_at = |o: usize| b.get(o..o + 2).map(|s| u16::from_le_bytes([s[0], s[1]]));
    let u32_at = |o: usize| b.get(o..o + 4).map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]));
    if b.get(0..4)? != b"\0PSF" {
        return None;
    }
    let key_table = u32_at(8)? as usize;
    let data_table = u32_at(12)? as usize;
    let count = u32_at(16)? as usize;
    let mut info = ParamInfo::default();
    let mut version = None;
    for i in 0..count.min(1024) {
        let e = 20 + i * 16;
        let key_off = key_table + u16_at(e)? as usize;
        let fmt = u16_at(e + 2)?;
        let len = u32_at(e + 4)? as usize;
        let data_off = data_table + u32_at(e + 12)? as usize;
        let key_end = b.get(key_off..)?.iter().position(|&c| c == 0)? + key_off;
        let key = std::str::from_utf8(&b[key_off..key_end]).ok()?;
        // 0x0004 utf8-special, 0x0204 utf8 (NUL-terminated); 0x0404 is int32.
        if fmt != 0x0204 && fmt != 0x0004 {
            continue;
        }
        let raw = b.get(data_off..data_off + len)?;
        let text = String::from_utf8_lossy(raw.split(|&c| c == 0).next().unwrap_or(raw)).into_owned();
        match key {
            "TITLE" => info.title = non_empty(Some(&text)),
            "TITLE_ID" => info.title_id = non_empty(Some(&text)),
            "APP_VER" => info.version = non_empty(Some(&text)),
            "VERSION" => version = non_empty(Some(&text)),
            "CATEGORY" => info.category = ContentCategory::from_sfo(&text),
            _ => {}
        }
    }
    if info.version.is_none() {
        info.version = version;
    }
    info.app_version = info.version.clone();
    Some(info)
}

/// A folder's `sce_sys/param.json`, else its `sce_sys/param.sfo`.
pub fn read_param(dir: &Path) -> Option<ParamInfo> {
    let sys = dir.join("sce_sys");
    std::fs::read_to_string(sys.join("param.json"))
        .ok()
        .and_then(|t| parse_param_json(&t))
        .or_else(|| std::fs::read(sys.join("param.sfo")).ok().and_then(|b| parse_param_sfo(&b)))
}

/// Read a game folder's metadata (folder name as the title fallback). Any
/// folder with an `eboot.bin` is accepted: this is for folders added by
/// hand; a scan uses [`classify_folder`].
pub fn read_game(dir: &Path) -> Option<Game> {
    let eboot = dir.join("eboot.bin");
    if !eboot.is_file() {
        return None;
    }
    Some(game_from(GameSource::Folder(dir.to_path_buf()), dir, eboot, read_param(dir).unwrap_or_default()))
}

/// Whether a folder found by a scan is a game: None without an
/// `eboot.bin`; else the game, or why it is not one (no param file, so no
/// complete game data; additional content; an update; another application
/// type).
pub fn classify_folder(dir: &Path) -> Option<Result<Game, String>> {
    let eboot = dir.join("eboot.bin");
    if !eboot.is_file() {
        return None;
    }
    let Some(info) = read_param(dir) else {
        return Some(Err("no sce_sys/param.json or param.sfo: not complete game data".into()));
    };
    if !info.category.is_game() {
        return Some(Err(info.category.describe()));
    }
    Some(Ok(game_from(GameSource::Folder(dir.to_path_buf()), dir, eboot, info)))
}

/// Extensions of an executable added by hand as a game.
pub const EXECUTABLE_EXTENSIONS: [&str; 3] = ["elf", "self", "bin"];

pub fn is_executable_file(path: &Path) -> bool {
    path.extension().is_some_and(|e| EXECUTABLE_EXTENSIONS.iter().any(|x| e.eq_ignore_ascii_case(x)))
}

/// A single executable added by hand, with the metadata of its folder when
/// that has a `sce_sys` (file name as the title fallback).
pub fn read_executable(path: &Path) -> Option<Game> {
    if !path.is_file() || !is_executable_file(path) {
        return None;
    }
    let dir = path.parent().map(Path::to_path_buf).unwrap_or_default();
    let mut info = read_param(&dir).unwrap_or_default();
    if info.title.is_none() {
        info.title = path.file_name().map(|n| n.to_string_lossy().into_owned());
    }
    Some(game_from(GameSource::Executable(path.to_path_buf()), &dir, path.to_path_buf(), info))
}

fn game_from(source: GameSource, dir: &Path, eboot: PathBuf, info: ParamInfo) -> Game {
    let icon = dir.join("sce_sys").join("icon0.png");
    let folder = dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    Game {
        source,
        dir: dir.to_path_buf(),
        eboot,
        title: info.title.unwrap_or(folder),
        title_id: info.title_id,
        version: info.version,
        app_version: info.app_version,
        firmware: info.firmware,
        icon: icon.is_file().then_some(icon),
        package: None,
    }
}

/// A title id (`AAAA00000`) inside `text`.
pub fn find_title_id(text: &str) -> Option<String> {
    let b = text.as_bytes();
    (0..b.len().saturating_sub(8)).find_map(|i| {
        let w = &b[i..i + 9];
        let ok = w[..4].iter().all(u8::is_ascii_uppercase) && w[4..].iter().all(u8::is_ascii_digit);
        let edge = |j: Option<&u8>| j.is_none_or(|c| !c.is_ascii_alphanumeric());
        (ok && edge(i.checked_sub(1).and_then(|j| b.get(j))) && edge(b.get(i + 9))).then(|| text[i..i + 9].to_string())
    })
}

/// Content id from a package header: magic `\x7FCNT`, the 36-byte content
/// id at 0x40 (`UP9000-PPSA01234_00-NAME...`).
pub fn package_content_id(header: &[u8]) -> Option<String> {
    if header.get(0..4)? != b"\x7FCNT" {
        return None;
    }
    let raw = header.get(0x40..0x64)?;
    let id = std::str::from_utf8(raw.split(|&c| c == 0).next()?).ok()?.trim().to_string();
    (!id.is_empty() && id.is_ascii()).then_some(id)
}

pub fn is_package_file(path: &Path) -> bool {
    path.extension().is_some_and(|e| PACKAGE_EXTENSIONS.iter().any(|x| e.eq_ignore_ascii_case(x)))
}

/// The state of a probed package.
pub fn package_state(info: &ps5pkg::PackageInfo) -> PackageState {
    match &info.status {
        ps5pkg::Status::Ready if info.kind == ps5pkg::ContentKind::Game => PackageState::Runnable,
        ps5pkg::Status::Ready => PackageState::NotAGame(format!("{} package: install it over the game", info.kind)),
        ps5pkg::Status::Encrypted(m) => PackageState::Encrypted(m.clone()),
        ps5pkg::Status::Unsupported(m) => PackageState::Unsupported(m.clone()),
    }
}

/// Name the library gives a package's icon.
pub fn package_icon_name(path: &Path) -> PathBuf {
    PathBuf::from(format!("{}#icon0.png", path.display()))
}

/// Read a package file's entry and its icon bytes: what the package
/// exposes (title, title id, version, icon, state), else the header's
/// content id gives the title id and the file name the title.
pub fn read_package_with_icon(path: &Path) -> Option<(Game, Option<Vec<u8>>)> {
    if !path.is_file() || !is_package_file(path) {
        return None;
    }
    let mut header = vec![0u8; 0x64];
    let n = std::fs::File::open(path).and_then(|mut f| std::io::Read::read(&mut f, &mut header)).unwrap_or(0);
    header.truncate(n);
    let content_id = package_content_id(&header);
    let stem = path.file_stem().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let mut g = Game {
        source: GameSource::Package(path.to_path_buf()),
        dir: path.parent().map(Path::to_path_buf).unwrap_or_default(),
        eboot: PathBuf::new(),
        title: stem.clone(),
        title_id: content_id.as_deref().and_then(find_title_id).or_else(|| find_title_id(&stem)),
        ..Default::default()
    };
    let mut icon = None;
    match ps5pkg::probe(path) {
        Ok(info) => {
            g.package = Some(package_state(&info));
            if let Some(t) = non_empty(info.name.as_deref()) {
                g.title = t;
            }
            if let Some(id) = non_empty(info.title_id.as_deref()) {
                g.title_id = Some(id);
            }
            g.version = non_empty(info.version.as_deref());
            g.app_version = g.version.clone();
            if let Some(bytes) = info.icon {
                g.icon = Some(package_icon_name(path));
                icon = Some(bytes);
            }
        }
        Err(e) => g.package = Some(PackageState::Unreadable(e.to_string())),
    }
    Some((g, icon))
}

/// Read a package file's entry (see [`read_package_with_icon`]).
pub fn read_package(path: &Path) -> Option<Game> {
    read_package_with_icon(path).map(|(g, _)| g)
}

/// Read one entry added by hand: a game folder, a package file or an
/// executable.
pub fn read_entry(path: &Path) -> Option<Game> {
    if path.is_dir() {
        read_game(path)
    } else if is_package_file(path) {
        read_package(path)
    } else {
        read_executable(path)
    }
}

/// Total size of the files below `path` (symlinks not followed).
pub fn disk_size(path: &Path) -> u64 {
    let Ok(meta) = std::fs::symlink_metadata(path) else { return 0 };
    if meta.is_file() {
        return meta.len();
    }
    let mut total = 0;
    let mut stack = vec![path.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            match e.file_type() {
                Ok(t) if t.is_dir() => stack.push(e.path()),
                Ok(t) if t.is_file() => total += e.metadata().map(|m| m.len()).unwrap_or(0),
                _ => {}
            }
        }
    }
    total
}

/// `1.5 GiB` style size text.
pub fn format_size(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = bytes as f64;
    let mut u = 0;
    while v >= 1024.0 && u + 1 < UNITS.len() {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 { format!("{bytes} B") } else { format!("{v:.2} {}", UNITS[u]) }
}

/// Find games under `roots`, plus the single `extra` entries.
///
/// A folder with an `eboot.bin` is not searched further; it is listed when
/// [`classify_folder`] finds a game in it. A package file is listed when it
/// is a runnable game package; loose executables are never listed (only as
/// `extra` entries). Hidden and symlinked folders are skipped. An `extra`
/// entry is listed whenever it can be read ([`read_entry`]). A game
/// reachable twice is listed once, and of the copies with one title id the
/// most complete is kept ([`dedup_by_title_id`]). Sorted by title.
pub fn scan(roots: &[PathBuf], extra: &[PathBuf]) -> Vec<Game> {
    scan_with_icons(roots, extra).0
}

/// [`scan`], plus the icons read from packages (by [`Game::icon`] name).
pub fn scan_with_icons(roots: &[PathBuf], extra: &[PathBuf]) -> (Vec<Game>, Vec<(PathBuf, Vec<u8>)>) {
    let mut found: Vec<(Game, Option<Vec<u8>>)> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut add = |path: &Path, entry: (Game, Option<Vec<u8>>), found: &mut Vec<(Game, Option<Vec<u8>>)>| {
        if seen.insert(std::fs::canonicalize(path).unwrap_or(path.to_path_buf())) {
            found.push(entry);
        }
    };
    for path in extra {
        let entry = if is_package_file(path) && path.is_file() {
            read_package_with_icon(path)
        } else {
            read_entry(path).map(|g| (g, None))
        };
        match entry {
            Some(e) => add(path, e, &mut found),
            None => log::debug!("library: cannot read {}", path.display()),
        }
    }
    for root in roots {
        let mut stack = vec![(root.clone(), 0usize)];
        while let Some((dir, depth)) = stack.pop() {
            match classify_folder(&dir) {
                Some(Ok(g)) => {
                    add(&dir, (g, None), &mut found);
                    continue;
                }
                Some(Err(why)) => {
                    log::debug!("library: {} is not a game: {why}", dir.display());
                    continue;
                }
                None => {}
            }
            if depth >= MAX_DEPTH {
                continue;
            }
            let Ok(rd) = std::fs::read_dir(&dir) else { continue };
            for e in rd.flatten() {
                let hidden = e.file_name().to_string_lossy().starts_with('.');
                let ft = e.file_type();
                let is_dir = ft.as_ref().map(|t| t.is_dir()).unwrap_or(false);
                let is_file = ft.as_ref().map(|t| t.is_file()).unwrap_or(false);
                if is_dir && !hidden {
                    stack.push((e.path(), depth + 1));
                } else if is_file && !hidden && is_package_file(&e.path()) {
                    match read_package_with_icon(&e.path()) {
                        Some(entry) if entry.0.package.as_ref().is_some_and(PackageState::can_run) => add(&e.path(), entry, &mut found),
                        Some((g, _)) => {
                            let state = g.package.as_ref().map(|p| format!("{} ({})", p.label(), p.detail())).unwrap_or_default();
                            log::debug!("library: {} is not a runnable game package: {state}", e.path().display());
                        }
                        None => {}
                    }
                }
            }
        }
    }
    let mut icon_bytes: std::collections::HashMap<PathBuf, Vec<u8>> = std::collections::HashMap::new();
    let mut games = Vec::with_capacity(found.len());
    for (g, icon) in found {
        if let (Some(name), Some(bytes)) = (g.icon.clone(), icon) {
            icon_bytes.insert(name, bytes);
        }
        games.push(g);
    }
    let mut sizes = std::collections::HashMap::new();
    let mut games = dedup_by_title_id(games, &mut |g: &Game| *sizes.entry(g.key()).or_insert_with(|| disk_size(g.source.path())));
    games.sort_by(|a, b| a.title.to_lowercase().cmp(&b.title.to_lowercase()).then(a.key().cmp(&b.key())));
    let icons = games.iter().filter_map(|g| g.icon.as_ref().and_then(|n| icon_bytes.remove_entry(n))).collect();
    (games, icons)
}

/// One entry per title id (compared without case): of the copies, the one
/// that runs, then a folder rather than a package, then one with an icon,
/// then the largest on disk (`size`, asked only for such ties) is kept;
/// on a full tie the first. Entries without a title id are all kept. The
/// order of the kept entries is that of their first copy.
pub fn dedup_by_title_id(games: Vec<Game>, size: &mut dyn FnMut(&Game) -> u64) -> Vec<Game> {
    let mut out: Vec<Game> = Vec::with_capacity(games.len());
    let mut by_id: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let rank = |g: &Game| (g.can_run(), !g.source.is_package(), g.icon.is_some());
    for g in games {
        let Some(id) = g.title_id.as_deref().map(str::to_ascii_uppercase) else {
            out.push(g);
            continue;
        };
        match by_id.get(&id) {
            None => {
                by_id.insert(id, out.len());
                out.push(g);
            }
            Some(&i) => {
                let (new, old) = (rank(&g), rank(&out[i]));
                let better = new > old || (new == old && size(&g) > size(&out[i]));
                log::debug!("library: {} and {} share title id {id}", out[i].key(), g.key());
                if better {
                    out[i] = g;
                }
            }
        }
    }
    out
}

/// Decode a PNG into RGBA8 (width, height, pixels).
pub fn load_png_rgba(path: &Path) -> Option<(u32, u32, Vec<u8>)> {
    let file = std::fs::File::open(path).ok()?;
    decode_png_rgba(std::io::BufReader::new(file))
}

/// Decode a PNG stream into RGBA8 (width, height, pixels).
pub fn decode_png_rgba<R: std::io::Read>(r: R) -> Option<(u32, u32, Vec<u8>)> {
    let mut dec = png::Decoder::new(r);
    dec.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = dec.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;
    buf.truncate(info.buffer_size());
    let (w, h) = (info.width, info.height);
    let rgba = match info.color_type {
        png::ColorType::Rgba => buf,
        png::ColorType::Rgb => buf.chunks_exact(3).flat_map(|p| [p[0], p[1], p[2], 255]).collect(),
        png::ColorType::GrayscaleAlpha => buf.chunks_exact(2).flat_map(|p| [p[0], p[0], p[0], p[1]]).collect(),
        png::ColorType::Grayscale => buf.iter().flat_map(|&g| [g, g, g, 255]).collect(),
        png::ColorType::Indexed => return None,
    };
    (rgba.len() == (w * h * 4) as usize).then_some((w, h, rgba))
}

/// The game's background picture (`sce_sys/pic0.png`, else `pic1.png`),
/// shrunk to at most `max_width` pixels wide.
pub fn load_banner(game_dir: &Path, max_width: u32) -> Option<(u32, u32, Vec<u8>)> {
    let sys = game_dir.join("sce_sys");
    let (w, h, rgba) = ["pic0.png", "pic1.png"].iter().find_map(|n| load_png_rgba(&sys.join(n)))?;
    Some(shrink(w, h, &rgba, w.div_ceil(max_width.max(1))))
}

/// Shrink an RGBA image by an integer `factor` (average of each block).
pub fn shrink(w: u32, h: u32, rgba: &[u8], factor: u32) -> (u32, u32, Vec<u8>) {
    let f = factor.max(1);
    if f == 1 {
        return (w, h, rgba.to_vec());
    }
    let (ow, oh) = ((w / f).max(1), (h / f).max(1));
    let mut out = Vec::with_capacity((ow * oh * 4) as usize);
    for y in 0..oh {
        for x in 0..ow {
            let mut sum = [0u32; 4];
            for dy in 0..f.min(h - y * f) {
                let row = ((y * f + dy) * w + x * f) as usize * 4;
                for dx in 0..f.min(w - x * f) as usize {
                    for c in 0..4 {
                        sum[c] += rgba[row + dx * 4 + c] as u32;
                    }
                }
            }
            let n = f.min(h - y * f) * f.min(w - x * f);
            out.extend(sum.map(|s| (s / n) as u8));
        }
    }
    (ow, oh, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shrink_averages_blocks() {
        // 4x2: left half black, right half white; halved to 2x1.
        let mut px = Vec::new();
        for _ in 0..2 {
            for x in 0..4 {
                let v = if x < 2 { 0 } else { 255 };
                px.extend([v, v, v, 255]);
            }
        }
        assert_eq!(shrink(4, 2, &px, 2), (2, 1, vec![0, 0, 0, 255, 255, 255, 255, 255]));
        assert_eq!(shrink(4, 2, &px, 1).2, px);
        // A factor that does not divide the size keeps whole blocks.
        assert_eq!(shrink(5, 3, &[9u8; 5 * 3 * 4], 2), (2, 1, vec![9; 8]));
    }

    #[test]
    fn param_json_fields() {
        let j = r#"{
            "titleId": "PPSA10112", "contentVersion": "01.000.001", "masterVersion": "01.00",
            "localizedParameters": {"defaultLanguage": "ja-JP",
              "en-US": {"titleName": "English"}, "ja-JP": {"titleName": "Japanese"}}
        }"#;
        let p = parse_param_json(j).unwrap();
        assert_eq!(p.title.as_deref(), Some("Japanese"));
        assert_eq!(p.title_id.as_deref(), Some("PPSA10112"));
        assert_eq!(p.version.as_deref(), Some("01.000.001"));
        assert_eq!(p.app_version.as_deref(), Some("01.000.001"));
        assert_eq!(p.firmware, None);
        let p = parse_param_json(r#"{"appVersion": "01.02", "contentVersion": "01.000.002", "requiredSystemSoftwareVersion": "0x1160000000000000"}"#).unwrap();
        assert_eq!((p.version.as_deref(), p.app_version.as_deref(), p.firmware.as_deref()), (Some("01.000.002"), Some("01.02"), Some("11.60")));
        let p = parse_param_json(r#"{"masterVersion": "01.00", "localizedParameters": {"fr-FR": {"titleName": "Jeu"}}}"#).unwrap();
        assert_eq!((p.title.as_deref(), p.version.as_deref(), p.title_id), (Some("Jeu"), Some("01.00"), None));
        assert!(parse_param_json("not json").is_none());
    }

    pub(crate) fn sfo(entries: &[(&str, &str)]) -> Vec<u8> {
        let mut keys = Vec::new();
        let mut data = Vec::new();
        let mut table = Vec::new();
        for (k, v) in entries {
            let ko = keys.len() as u16;
            keys.extend_from_slice(k.as_bytes());
            keys.push(0);
            let doff = data.len() as u32;
            let mut val = v.as_bytes().to_vec();
            val.push(0);
            let len = val.len() as u32;
            let max = (len + 3) & !3;
            val.resize(max as usize, 0);
            data.extend_from_slice(&val);
            table.extend_from_slice(&ko.to_le_bytes());
            table.extend_from_slice(&0x0204u16.to_le_bytes());
            table.extend_from_slice(&len.to_le_bytes());
            table.extend_from_slice(&max.to_le_bytes());
            table.extend_from_slice(&doff.to_le_bytes());
        }
        while keys.len() % 4 != 0 {
            keys.push(0);
        }
        let key_table = 20 + table.len() as u32;
        let data_table = key_table + keys.len() as u32;
        let mut b = b"\0PSF".to_vec();
        b.extend_from_slice(&0x101u32.to_le_bytes());
        b.extend_from_slice(&key_table.to_le_bytes());
        b.extend_from_slice(&data_table.to_le_bytes());
        b.extend_from_slice(&(entries.len() as u32).to_le_bytes());
        b.extend(table);
        b.extend(keys);
        b.extend(data);
        b
    }

    #[test]
    fn param_sfo_fields() {
        let b = sfo(&[("APP_VER", "01.02"), ("TITLE", "Some Game"), ("TITLE_ID", "CUSA00001"), ("VERSION", "01.00")]);
        let p = parse_param_sfo(&b).unwrap();
        assert_eq!(p.title.as_deref(), Some("Some Game"));
        assert_eq!(p.title_id.as_deref(), Some("CUSA00001"));
        assert_eq!(p.version.as_deref(), Some("01.02"));
        let p = parse_param_sfo(&sfo(&[("VERSION", "01.00")])).unwrap();
        assert_eq!(p.version.as_deref(), Some("01.00"));
        assert!(parse_param_sfo(b"nope").is_none());
        assert!(parse_param_sfo(&b[..30]).is_none());
    }

    #[test]
    fn helpers() {
        assert_eq!(firmware_version("0x0820000000000000").as_deref(), Some("8.20"));
        assert_eq!(firmware_version("0x1160010000000000").as_deref(), Some("11.60.01"));
        assert_eq!(firmware_version("0x116"), None);
        assert_eq!(find_title_id("UP9000-PPSA03016_00-X").as_deref(), Some("PPSA03016"));
        assert_eq!(find_title_id("SILENT HILL PPSA10112-app").as_deref(), Some("PPSA10112"));
        assert_eq!(find_title_id("XPPSA10112"), None);
        assert_eq!(format_size(512), "512 B");
        assert_eq!(format_size(3 * 1024 * 1024 / 2), "1.50 MiB");
        let g = Game { eboot: "/g/eboot.bin".into(), dir: "/g".into(), ..Default::default() };
        assert_eq!(g.executable("eboot.bin"), PathBuf::from("/g/eboot.bin"));
        assert_eq!(g.executable("other.elf"), PathBuf::from("/g/other.elf"));
    }

    #[test]
    fn scan_finds_nested_games_once() {
        let root = crate::test_dir().join(format!("scan-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let g1 = root.join("a/GameOne");
        let g2 = root.join("b/c/GameTwo");
        let hidden = root.join(".hidden/G");
        for d in [&g1, &g2, &hidden] {
            std::fs::create_dir_all(d.join("sce_sys")).unwrap();
            std::fs::write(d.join("eboot.bin"), b"x").unwrap();
        }
        std::fs::write(g1.join("sce_sys/param.json"), r#"{"titleId":"PPSA00001","applicationCategoryType":0,"localizedParameters":{"en-US":{"titleName":"Zeta"}}}"#).unwrap();
        std::fs::write(g2.join("sce_sys/param.sfo"), sfo(&[("CATEGORY", "gd")])).unwrap();
        // A folder inside a game is not searched.
        std::fs::create_dir_all(g2.join("sub")).unwrap();
        std::fs::write(g2.join("sub/eboot.bin"), b"x").unwrap();
        // A package next to the games, and one added by hand.
        let mut pkg = b"\x7FCNT".to_vec();
        pkg.resize(0x40, 0);
        pkg.extend_from_slice(b"UP9000-PPSA03016_00-MARVELSPIDERMAN2\0");
        std::fs::write(root.join("b/game.pkg"), &pkg).unwrap();
        let extra = root.join("elsewhere/Some Game [PPSA12345].ffpkg");
        std::fs::create_dir_all(extra.parent().unwrap()).unwrap();
        std::fs::write(&extra, b"not a cnt header").unwrap();
        let games = scan(&[root.clone(), root.join("a")], &[extra.clone(), root.join("b/game.pkg")]);
        let titles: Vec<&str> = games.iter().map(|g| g.title.as_str()).collect();
        assert_eq!(titles, ["game", "GameTwo", "Some Game [PPSA12345]", "Zeta"]);
        assert_eq!(games[3].title_id.as_deref(), Some("PPSA00001"));
        assert!(games[1].icon.is_none());
        assert_eq!(games[0].source, GameSource::Package(root.join("b/game.pkg")));
        assert_eq!(games[0].title_id.as_deref(), Some("PPSA03016"));
        assert_eq!(games[2].title_id.as_deref(), Some("PPSA12345"));
        assert!(games[2].source.is_package() && !games[1].source.is_package());
        assert_eq!(disk_size(&root.join("b/game.pkg")), pkg.len() as u64);
        assert!(disk_size(&root) >= pkg.len() as u64 + 3);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Prints what a scan of `$RPS5E_GUI_SCAN_DIR` lists (read-only):
    /// `RPS5E_GUI_SCAN_DIR=~/games cargo test -p rps5e-gui print_scan -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn print_scan() {
        let Some(dir) = std::env::var_os("RPS5E_GUI_SCAN_DIR") else { return };
        for g in scan(&[PathBuf::from(dir)], &[]) {
            let state = g.package.as_ref().map(|p| p.label()).unwrap_or("folder");
            println!("{} | {} | {} | {}", g.title, g.title_id.as_deref().unwrap_or("-"), state, g.key());
        }
    }

    /// Real game folders found by title ID under `GAMES_DIR` (each skipped
    /// when absent).
    #[test]
    fn real_game_folders() {
        let cases = [
            ("PPSA10112", "SILENT HILL: The Short Message", "01.000.001"),
            ("PPSA25872", "Terminator 2D: NO FATE", "01.000.000"),
            ("PPSA21567", "ASTRO BOT", "01.018.000"),
        ];
        for (id, title, version) in cases {
            let Some(dir) = ps5vm::test_games::game_dir(id) else {
                eprintln!("skipping {id} (not present)");
                continue;
            };
            let dir = dir.as_path();
            let g = read_game(dir).unwrap();
            assert_eq!(g.title_id.as_deref(), Some(id));
            assert_eq!(g.title, title);
            assert_eq!(g.version.as_deref(), Some(version));
            let icon = g.icon.as_ref().expect("icon0.png");
            let (w, h, px) = load_png_rgba(icon).expect("icon decodes");
            assert!(w >= 64 && h >= 64 && px.len() == (w * h * 4) as usize);
            // Scanning the parent finds it.
            assert!(g.firmware.is_some());
            let found = scan(&[dir.parent().unwrap().to_path_buf()], &[]);
            assert!(found.iter().any(|f| f.title_id.as_deref() == Some(id)));
            // The background picture of the details panel, shrunk.
            if dir.join("sce_sys/pic0.png").is_file() {
                let (w, h, px) = load_banner(dir, 720).expect("pic0.png decodes");
                assert!(w <= 720 && w > 300 && h > 100 && px.len() == (w * h * 4) as usize, "{w}x{h}");
            }
        }
    }
}

#[cfg(test)]
mod scan_tests {
    use super::*;

    /// A scratch tree under the target directory, removed on drop.
    struct Tree(PathBuf);

    impl Tree {
        fn new(name: &str) -> Self {
            let root = crate::test_dir().join(format!("{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&root);
            std::fs::create_dir_all(&root).unwrap();
            Tree(root)
        }

        fn file(&self, rel: &str, bytes: &[u8]) -> PathBuf {
            let p = self.0.join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(&p, bytes).unwrap();
            p
        }

        /// `rel/sce_sys/param.json` with `category` (None: no
        /// `applicationCategoryType`, as for additional content).
        fn param(&self, rel: &str, id: &str, title: &str, category: Option<u32>) {
            let cat = category.map(|c| format!(r#""applicationCategoryType":{c},"#)).unwrap_or_default();
            let json = format!(r#"{{{cat}"titleId":"{id}","contentVersion":"01.000.000","localizedParameters":{{"en-US":{{"titleName":"{title}"}}}}}}"#);
            self.file(&format!("{rel}/sce_sys/param.json"), json.as_bytes());
        }

        fn game(&self, rel: &str, id: &str, title: &str, payload: usize) {
            self.file(&format!("{rel}/eboot.bin"), b"\x7fELF");
            self.param(rel, id, title, Some(0));
            self.file(&format!("{rel}/data.bin"), &vec![0u8; payload]);
        }
    }

    impl Drop for Tree {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn keys(games: &[Game], root: &Path) -> Vec<String> {
        games.iter().map(|g| g.source.path().strip_prefix(root).unwrap().display().to_string()).collect()
    }

    #[test]
    fn categories() {
        let cat = |j: &str| parse_param_json(j).unwrap().category;
        assert_eq!(cat(r#"{"applicationCategoryType":0}"#), ContentCategory::Game);
        assert_eq!(cat(r#"{"titleId":"PPSA00001"}"#), ContentCategory::AdditionalContent);
        assert!(matches!(cat(r#"{"applicationCategoryType":33554432}"#), ContentCategory::Other(_)));
        assert!(!cat(r#"{"applicationCategoryType":65536}"#).is_game());
        let sfo_cat = |c: &str| ContentCategory::from_sfo(c);
        assert!(sfo_cat("gd").is_game() && sfo_cat("gde").is_game());
        assert_eq!(sfo_cat("gp"), ContentCategory::Patch);
        assert_eq!(sfo_cat("ac"), ContentCategory::AdditionalContent);
        assert!(!sfo_cat("sd").is_game());
        // A param.sfo without CATEGORY: taken as a game.
        assert!(ContentCategory::default().is_game());
    }

    /// The layout of a typical release folder: the game, fixes, region
    /// swaps, a backport with a loose ELF and archives, DLC packages.
    #[test]
    fn release_folder_lists_only_the_game() {
        let t = Tree::new("gta");
        let rel = "[PS5] Grand Theft Auto V (2022) (PPSA04263) (EU) v1.010.002";
        t.game(&format!("{rel}/PPSA04263"), "PPSA04263", "Grand Theft Auto V", 64);
        t.file(&format!("{rel}/PPSA04263/sce_sys/icon0.png"), b"png");
        t.file(&format!("{rel}/Fix/Fix 1/fakelib/libSceNpAuth.sprx"), b"x");
        t.file(&format!("{rel}/Fix/Fix 2/fakelib/libSceNpAuth.sprx"), b"x");
        // Region swaps: only sce_sys, no eboot.bin.
        t.param(&format!("{rel}/Optional/EU_to_US"), "PPSA04264", "Grand Theft Auto V", Some(0));
        t.param(&format!("{rel}/Optional/Original EU"), "PPSA04263", "Grand Theft Auto V", Some(0));
        let bp = format!("{rel}/[4.xx+] [Ciss84] Backport(backpork) for v1.010.002");
        t.file(&format!("{bp}/ps5-backpork.elf"), b"\x7fELF");
        t.file(&format!("{bp}/[4.xx+] Backport(backpork) for v1.010.002.7z"), b"7z");
        t.file(&format!("{bp}/Original_files for v1.010.002.7z"), b"7z");
        // Retail-encrypted DLC packages.
        t.file(&format!("{rel}/DLC/EU (PPSA04263)/EP1004-PPSA04263_00-GTAVBONUS0000000.pkg"), b"\x7fFIH\x01\x00\x03\x00");
        t.file(&format!("{rel}/DLC/US (PPSA04264)/UP1004-PPSA04264_00-GTAVBONUS0000000.pkg"), b"\x7fFIH\x01\x00\x03\x00");
        let games = scan(&[t.0.clone()], &[]);
        assert_eq!(keys(&games, &t.0), [format!("{rel}/PPSA04263")]);
        let g = &games[0];
        assert_eq!((g.title.as_str(), g.title_id.as_deref()), ("Grand Theft Auto V", Some("PPSA04263")));
        assert!(g.can_run() && g.icon.is_some());
    }

    #[test]
    fn dlc_patch_and_incomplete_folders_are_not_games() {
        let t = Tree::new("notgames");
        t.game("Game", "PPSA00010", "Game", 8);
        // Additional content: param.json without applicationCategoryType.
        t.file("DLC/eboot.bin", b"x");
        t.param("DLC", "PPSA00010", "Game DLC", None);
        // A system application.
        t.file("SysApp/eboot.bin", b"x");
        t.param("SysApp", "NPXS40000", "System", Some(33554432));
        // An update dumped as a PS4-style folder.
        t.file("Patch/eboot.bin", b"x");
        t.file("Patch/sce_sys/param.sfo", &super::tests::sfo(&[("CATEGORY", "gp"), ("TITLE_ID", "CUSA00020")]));
        // A fix: an executable without param.json.
        t.file("Fix/eboot.bin", b"x");
        // A loose executable and a module next to the games.
        t.file("payload.elf", b"\x7fELF");
        t.file("tool/thing.self", b"x");
        let games = scan(&[t.0.clone()], &[]);
        assert_eq!(keys(&games, &t.0), ["Game"]);
        let why = |rel: &str| classify_folder(&t.0.join(rel)).unwrap().unwrap_err();
        assert_eq!(why("DLC"), "additional content (DLC)");
        assert_eq!(why("Patch"), "update");
        assert!(why("Fix").contains("not complete game data"));
        assert!(why("SysApp").starts_with("not a game application"));
        assert!(classify_folder(&t.0.join("tool")).is_none());
        // Added by hand, a folder with eboot.bin and an executable are kept.
        let games = scan(&[t.0.clone()], &[t.0.join("Fix"), t.0.join("payload.elf"), t.0.join("missing.elf")]);
        assert_eq!(keys(&games, &t.0), ["Fix", "Game", "payload.elf"]);
        let elf = &games[2];
        assert_eq!(elf.source, GameSource::Executable(t.0.join("payload.elf")));
        assert_eq!(elf.executable("eboot.bin"), t.0.join("payload.elf"));
        assert_eq!(elf.executable("other.elf"), t.0.join("payload.elf"));
        assert!(elf.can_run() && elf.title == "payload.elf");
        assert!(read_entry(&t.0.join("payload.elf")).is_some());
        assert!(read_entry(&t.0.join("Game/sce_sys/param.json")).is_none());
    }

    #[test]
    fn duplicates_keep_the_most_complete_copy() {
        let t = Tree::new("dedup");
        // The same title twice: a full copy and a small leftover.
        t.game("Full/PPSA00030", "PPSA00030", "Thirty", 4096);
        t.game("Partial/PPSA00030", "ppsa00030", "Thirty", 16);
        // Copies with an icon win over larger ones without.
        t.game("A/PPSA00040", "PPSA00040", "Forty", 4096);
        t.game("B/PPSA00040", "PPSA00040", "Forty", 16);
        t.file("B/PPSA00040/sce_sys/icon0.png", b"png");
        // Different title ids stay apart.
        t.game("C/PPSA00050", "PPSA00050", "Forty", 16);
        let games = scan(&[t.0.clone()], &[]);
        assert_eq!(keys(&games, &t.0), ["B/PPSA00040", "C/PPSA00050", "Full/PPSA00030"]);
        // The same folder reachable twice (two roots, and added by hand).
        let games = scan(&[t.0.clone(), t.0.join("Full")], &[t.0.join("Full/PPSA00030")]);
        assert_eq!(games.len(), 3);
        // A folder wins over a package; entries without a title id stay.
        let folder = Game { source: GameSource::Folder("/f".into()), title_id: Some("PPSA1".into()), ..Default::default() };
        let pkg = Game {
            source: GameSource::Package("/p.pkg".into()),
            title_id: Some("PPSA1".into()),
            package: Some(PackageState::Runnable),
            icon: Some("/p.pkg#icon0.png".into()),
            ..Default::default()
        };
        let none = Game { source: GameSource::Folder("/n".into()), ..Default::default() };
        let mut asked = 0;
        let out = dedup_by_title_id(vec![pkg.clone(), none.clone(), folder.clone(), none.clone()], &mut |_| {
            asked += 1;
            0
        });
        assert_eq!(out, [folder, none.clone(), none]);
        assert_eq!(asked, 0, "sizes are read only for ties");
    }
}

#[cfg(test)]
mod package_tests {
    use super::*;
    use ps5pkg::build::{fake_pkg_image, PfsBuilder, PkgBuilder};
    use ps5pkg::param::{build_sfo, SfoValue};

    const CID: &str = "UP9000-PPSA99998_00-LIBRARYTEST00000";

    fn package(path: &Path, image_passcode: &str, header_passcode: &str, category: &str, icon: Option<Vec<u8>>) {
        let json = r#"{"titleId":"PPSA99998","contentVersion":"01.000.002",
            "localizedParameters":{"defaultLanguage":"en-US","en-US":{"titleName":"Library Test"}}}"#;
        let tree = PfsBuilder::new().file("eboot.bin", b"\x7fELF").file("sce_sys/param.json", json.as_bytes());
        let mut b = PkgBuilder::new(CID, fake_pkg_image(&tree, CID, image_passcode, false));
        b.passcode = Some(header_passcode.to_string());
        b.param_sfo = Some(build_sfo(&[
            ("CATEGORY", SfoValue::Str(category.into())),
            ("TITLE_ID", SfoValue::Str("PPSA99998".into())),
        ]));
        b.icon0 = icon;
        std::fs::write(path, b.build()).unwrap();
    }

    fn tiny_png() -> Vec<u8> {
        let mut out = Vec::new();
        let mut enc = png::Encoder::new(&mut out, 2, 2);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        enc.write_header().unwrap().write_image_data(&[200u8; 16]).unwrap();
        out
    }

    #[test]
    fn package_states_title_and_icon() {
        let root = crate::test_dir().join(format!("pkgstate-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("nested/deeper")).unwrap();
        let zeros = "0".repeat(32);
        // A fake game package: runnable, with what it holds.
        let fake = root.join("nested/deeper/game.pkg");
        package(&fake, &zeros, &zeros, "gd", Some(tiny_png()));
        // Its image keyed with another passcode: the PFS key cannot be
        // derived, as for a retail package.
        let locked = root.join("locked.pkg");
        package(&locked, &"7".repeat(32), &"7".repeat(32), "gd", None);
        // An update package.
        let update = root.join("update.pkg");
        package(&update, &zeros, &zeros, "gp", None);
        // A folder scan lists only the runnable game package.
        let (games, _) = scan_with_icons(&[root.clone()], &[]);
        assert_eq!(games.iter().map(|g| g.source.path()).collect::<Vec<_>>(), [fake.as_path()]);
        // Added by hand, the others are read with their state; all three
        // have one title id, so the runnable one is the card shown.
        let (games, icons) = scan_with_icons(&[root.clone()], &[locked.clone(), update.clone()]);
        assert_eq!(games.len(), 1);
        let g = &games[0];
        assert_eq!(g.package, Some(PackageState::Runnable));
        assert!(g.can_run());
        assert_eq!(g.title, "Library Test");
        assert_eq!(g.title_id.as_deref(), Some("PPSA99998"));
        assert_eq!(g.version.as_deref(), Some("01.000.002"));
        assert_eq!(g.icon, Some(package_icon_name(&fake)));
        let (_, bytes) = icons.iter().find(|(n, _)| *n == package_icon_name(&fake)).expect("icon bytes");
        assert_eq!(decode_png_rgba(&bytes[..]).map(|(w, h, _)| (w, h)), Some((2, 2)));
        let l = read_package(&locked).unwrap();
        assert!(matches!(l.package, Some(PackageState::Encrypted(_))), "{:?}", l.package);
        assert!(!l.can_run());
        assert_eq!(l.package.as_ref().unwrap().label(), "encrypted, cannot be run");
        let u = read_package(&update).unwrap();
        // The locked package alone, added by hand, is listed.
        let (games, _) = scan_with_icons(&[], &[locked.clone()]);
        assert_eq!(games.iter().map(|g| g.source.path()).collect::<Vec<_>>(), [locked.as_path()]);
        assert!(matches!(u.package, Some(PackageState::NotAGame(_))), "{:?}", u.package);
        assert!(!u.can_run());
        // Game folders are always runnable.
        assert!(Game::default().can_run());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn disk_images_are_listed_as_games() {
        use ps5pkg::build::{ExFatBuilder, PfsBuilder};
        let root = crate::test_dir().join(format!("diskimg-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let json = r#"{"titleId":"PPSA99997","contentVersion":"01.003.000",
            "localizedParameters":{"defaultLanguage":"en-US","en-US":{"titleName":"Image Test"}}}"#;
        let vol = ExFatBuilder::new()
            .file("PPSA99997-app/eboot.bin", b"\x7fELF")
            .file("PPSA99997-app/sce_sys/param.json", json.as_bytes())
            .file("PPSA99997-app/sce_sys/icon0.png", &tiny_png())
            .build();
        let exfat = root.join("Image Test.EXFAT");
        vol.write_to(&exfat).unwrap();
        let container = root.join("sub/image.ffpfsc");
        std::fs::create_dir_all(container.parent().unwrap()).unwrap();
        std::fs::write(&container, PfsBuilder::new().compressed_file("game.exfat", &vol.to_vec()).build()).unwrap();
        assert!(is_package_file(&exfat) && is_package_file(&container));
        for path in [&exfat, &container] {
            let (g, icon) = read_package_with_icon(path).unwrap();
            assert_eq!(g.package, Some(PackageState::Runnable), "{}", path.display());
            assert_eq!(g.title, "Image Test");
            assert_eq!(g.title_id.as_deref(), Some("PPSA99997"));
            assert_eq!(g.version.as_deref(), Some("01.003.000"));
            assert_eq!(icon, Some(tiny_png()));
        }
        // One card per title id.
        let (games, _) = scan_with_icons(&[root.clone()], &[]);
        assert_eq!(games.len(), 1);
        assert!(games[0].source.is_package());
        let _ = std::fs::remove_dir_all(&root);
    }
}
